"""
LangGraph 全量改造 · 无LLM验证脚本（不连数据库/LLM，随时可跑）
用假 planner + InMemorySaver 驱动【真实编译的图】（YAML流程→子图），覆盖：
  1. 图编译与拓扑（主图 + 流程子图节点）
  2. 业务流程 interrupt 槽位收集：查订单 → 中断问订单号 → 用户回答恢复 → 流程走完
  3. 多意图拒绝 → 挂起意图压栈 → 执行选中流程 → end 后追问
  4. 卡片直填槽位（对象消息免提问）
  5. checkpointer 持久化：同 thread 重建 DialogueApp 后状态/中断点不丢
运行：uv run python -m atguigu.test.langgraph_full_check
"""

import asyncio

from langgraph.checkpoint.memory import MemorySaver

from atguigu.domain.messages import UserMessage, MessageType, FocusedObject
from atguigu.plan.turn_plan import (TaskTurnPlan, KnowledgeTurnPlan, ChitChatTurnPlan, TurnPlan)
from atguigu.plan.commands import ResumeFlowCommand, StartFlowCommand, SetSlotsCommand
from atguigu.task.flows.loader import FlowLoader
from atguigu.task.action.buidler import build_action_runner
from atguigu.graph.main_graph import build_main_graph
from atguigu.graph.app import DialogueApp
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def check(cond: bool, tip: str):
    assert cond, f"FAIL: {tip}"
    print(f"  ✓ {tip}")


_message_seq = 0


def _next_message_id() -> str:
    global _message_seq
    _message_seq += 1
    return f"m{_message_seq}"


def text_msg(text: str, sender="u1") -> UserMessage:
    return UserMessage(
        sender_id=sender,
        message_id=_next_message_id(),
        type=MessageType.TEXT,
        text=text,
    )


def card_msg(obj_id="A001", obj_type="order", sender="u1") -> UserMessage:
    return UserMessage(
        sender_id=sender,
        message_id=_next_message_id(),
        type=MessageType.OBJECT,
        object=FocusedObject(id=obj_id, type=obj_type, title=obj_id, attributes={}),
    )


# ============================== 假组件 ==============================

class FakePlanner:
    def __init__(self, plans: list[TurnPlan]):
        self._plans = list(plans)

    async def predict(self, ctx, flows_list, knowledge_intents, **kwargs) -> TurnPlan:
        if self._plans:
            return self._plans.pop(0)

        # collect interrupt 恢复后现在也会过 Planner。测试里没有显式脚本时，
        # 按当前 step 把用户文本当作对应槽位，模拟正常“回答当前问题”。
        step_to_slot = {
            "ask_order_number": "order_number",
            "ask_refund_reason": "refund_reason",
            "ask_product_id": "product_id",
        }
        step_id = kwargs.get("active_flow_step")
        slot_name = step_to_slot.get(step_id)
        if slot_name:
            return TurnPlan(task=TaskTurnPlan(commands=[
                SetSlotsCommand(
                    command="set_slots",
                    slots={slot_name: ctx.user_message_text()},
                )
            ]))

        raise AssertionError(f"FakePlanner 缺少 TurnPlan 脚本，step={step_id}")


class FakeClarifyResponder:
    async def respond(self, reason, ctx):
        from atguigu.domain.messages import BotMessage
        return [BotMessage(text=f"[澄清:{reason.value}]")]


class FakeKnowledgeHandler:
    from atguigu.knowledge.intents import KNOWLEDGE_INTENTS as knowledge_intents

    async def handle(self, ctx, intents):
        from atguigu.domain.messages import BotMessage
        return [BotMessage(text=f"[知识:{','.join(intents)}]")]


class FakeChitChatHandler:
    async def handle(self, chat, ctx):
        from atguigu.domain.messages import BotMessage
        return [BotMessage(text=f"[闲聊:{chat}]")]


def make_saver() -> MemorySaver:
    """带领域类型 allowlist 的内存 checkpointer（消除反序列化告警）"""
    from atguigu.graph.builder import build_checkpoint_serde
    return MemorySaver(serde=build_checkpoint_serde())


def build_app(checkpointer, plans: list[TurnPlan]) -> DialogueApp:
    flows_list = FlowLoader().load_multi_yaml([
        PROJECT_ROOT / "flow_config" / "system_flows.yml",
        PROJECT_ROOT / "flow_config" / "user_flows.yml",
    ])
    graph = build_main_graph(
        planner=FakePlanner(plans),
        validator=__import__("atguigu.plan.validator", fromlist=["TurnPlanValidator"]).TurnPlanValidator(),
        clarify_responder=FakeClarifyResponder(),
        knowledge_handler=FakeKnowledgeHandler(),
        chitchat_handler=FakeChitChatHandler(),
        flows_list=flows_list,
        action_runner=build_action_runner(),
        checkpointer=checkpointer,
    )
    return DialogueApp(graph, checkpointer)


# ============================== 用例 ==============================

async def test_topology():
    print("[1] 图编译与拓扑")
    app = build_app(make_saver(), [])
    nodes = set(app._graph.get_graph().nodes.keys())
    for expected in ("ingest", "plan", "validate", "task", "interrupt_dispatch",
                     "flow_exit", "knowledge", "chitchat",
                     "clarify", "object_dispatch", "pending_follow_up",
                     "flow_order_status_query", "flow_logistics_tracking", "flow_refund_request"):
        check(expected in nodes, f"节点存在: {expected}")


async def test_flow_with_interrupt_collect():
    print("[2] 业务流程 + interrupt 槽位收集（查订单全流程）")
    plans = [TurnPlan(task=TaskTurnPlan(commands=[
        StartFlowCommand(command="start_flow", flow="order_status_query"),
    ]))]
    app = build_app(make_saver(), plans)

    # 轮1：开流程 → collect 订单号 → interrupt 暂停，机器人在本轮发出"请告诉我你的订单号"
    r1 = await app.chat(text_msg("查下订单"))
    texts1 = [m.text for m in r1.messages]
    check(any("订单状态查询" in t for t in texts1), "轮1: 开流程过场话术（flow_context 渲染）")
    check(any("订单号" in t for t in texts1), "轮1: collect 中断并发出订单号提问")

    # 图应处于中断点
    snap = await app._graph.aget_state({"configurable": {"thread_id": "u1"}})
    check(bool(snap.next), "图停在中断点（interrupt 生效）")

    # 轮2：用户回答订单号 → 恢复 → 流程走完（订单查询失败也不影响断言）
    r2 = await app.chat(text_msg("A001"))
    texts2 = [m.text for m in r2.messages]
    check(any("订单A001当前状态" in t for t in texts2), "轮2: 槽位收集完成，流程执行到响应渲染")

    snap2 = await app._graph.aget_state({"configurable": {"thread_id": "u1"}})
    check(snap2.values.get("active_flow") is None, "流程结束（active_flow 清空）")
    check(snap2.values.get("slots", {}).get("order_number") == "A001", "订单号已入槽位")

    history = await app.get_history("u1")
    user_texts = [item["text"] for item in history if item["role"] == "user"]
    check("查下订单" in user_texts, "历史保留发起流程的用户话术")
    check("A001" in user_texts, "interrupt 恢复答案写入持久化历史")


async def test_multi_intent_pending():
    print("[3] 多意图 → 挂起意图 → 完成后追问")
    plans = [
        # 轮1：两个 start_flow → MULTIPLE_TASK_FLOWS 拒绝 → 压栈 → 澄清
        TurnPlan(task=TaskTurnPlan(commands=[
            StartFlowCommand(command="start_flow", flow="order_status_query"),
            StartFlowCommand(command="start_flow", flow="refund_request"),
            SetSlotsCommand(command="set_slots", slots={"order_number": "A001"}),
        ])),
        # 轮2：用户选查订单（合法）→ 执行 → interrupt 收集…不，订单号这轮已带 slot？
        # task 节点 set_slots 先填槽，子图 collect 发现已有值直接放行 → 流程走完 → 追问
        TurnPlan(task=TaskTurnPlan(commands=[
            StartFlowCommand(command="start_flow", flow="order_status_query"),
            SetSlotsCommand(command="set_slots", slots={"order_number": "A001"}),
        ])),
    ]
    app = build_app(make_saver(), plans)

    r1 = await app.chat(text_msg("查订单，没发货就退货"))
    check(any("[澄清:multiple_task_flows]" in m.text for m in r1.messages), "轮1: 多意图被拒走澄清")
    snap = await app._graph.aget_state({"configurable": {"thread_id": "u1"}})
    check(any("refund_request" in e["flows"] for e in snap.values.get("pending_intents", [])),
          "轮1: refund_request 压入挂起意图队列")

    r2 = await app.chat(text_msg("先查订单"))
    texts2 = [m.text for m in r2.messages]
    check(any("订单A001当前状态" in t for t in texts2), "轮2: 选中流程执行完成（槽位来自set_slots）")
    check(any("售后/退款建议" in t and "继续帮你处理" in t for t in texts2), "轮2: 流程end后追问挂起的售后意图")
    snap2 = await app._graph.aget_state({"configurable": {"thread_id": "u1"}})
    check(snap2.values.get("pending_intents") in ([], None), "追问后挂起队列清空")


async def test_card_direct_fill():
    print("[4] 卡片直填槽位（对象消息免提问）")
    plans = [TurnPlan(task=TaskTurnPlan(commands=[
        StartFlowCommand(command="start_flow", flow="order_status_query"),
    ]))]
    app = build_app(make_saver(), plans)

    # 轮1：开流程 → interrupt 等订单号
    await app.chat(text_msg("查订单"))
    # 轮2：用户点订单卡片 → ingest 存 focused_object → 子图恢复 → collect 卡片直填 → 走完
    r2 = await app.chat(card_msg("A202"))
    texts2 = [m.text for m in r2.messages]
    check(any("订单A202当前状态" in t for t in texts2), "卡片直填订单号，流程免提问直接走完")
    snap = await app._graph.aget_state({"configurable": {"thread_id": "u1"}})
    check(snap.values.get("slots", {}).get("order_number") == "A202", "卡片ID写入槽位")


async def test_checkpointer_persistence():
    print("[5] checkpointer 持久化：中断点跨实例恢复")
    plans = [TurnPlan(task=TaskTurnPlan(commands=[
        StartFlowCommand(command="start_flow", flow="order_status_query"),
    ]))]
    saver = make_saver()
    app1 = build_app(saver, plans)

    # 轮1：开流程 → 中断等订单号
    await app1.chat(text_msg("查订单", sender="u9"))
    snap = await saver.aget_tuple({"configurable": {"thread_id": "u9"}})
    check(snap is not None, "checkpoint 已写入")

    # 模拟进程重启：用同一个 saver 构建全新 DialogueApp（图重新编译）
    app2 = build_app(saver, [])  # planner空脚本：恢复路径不再调planner
    r = await app2.chat(text_msg("A999", sender="u9"))
    texts = [m.text for m in r.messages]
    check(any("订单A999当前状态" in t for t in texts), "重启后从中断点恢复，用户回答直达流程")


async def test_interrupt_switch_and_precise_resume():
    print("[6] collect 中切换意图 + 指定 Flow 精确恢复")
    plans = [
        # 轮1：退款流程已带订单号，直接停在 ask_refund_reason。
        TurnPlan(task=TaskTurnPlan(commands=[
            StartFlowCommand(command="start_flow", flow="refund_request"),
            SetSlotsCommand(command="set_slots", slots={"order_number": "A001"}),
        ])),
        # 轮2：用户没有回答退款原因，而是在 interrupt 中改查另一张订单。
        TurnPlan(task=TaskTurnPlan(commands=[
            StartFlowCommand(command="start_flow", flow="order_status_query"),
            SetSlotsCommand(command="set_slots", slots={"order_number": "B002"}),
        ])),
        # 轮3：明确恢复之前暂停的退款 Flow。
        TurnPlan(task=TaskTurnPlan(commands=[
            ResumeFlowCommand(command="resume_flow", flow="refund_request"),
        ])),
    ]
    app = build_app(make_saver(), plans)
    config = {"configurable": {"thread_id": "switch-u"}}

    r1 = await app.chat(text_msg("订单A001要退款", sender="switch-u"))
    check(any("退款原因" in m.text for m in r1.messages), "轮1: 退款流程停在退款原因 collect")

    r2 = await app.chat(text_msg("先查订单B002的状态", sender="switch-u"))
    texts2 = [m.text for m in r2.messages]
    check(any("放一放" in t for t in texts2), "轮2: interrupt 输入识别为新意图并暂停旧 Flow")
    check(any("订单B002当前状态" in t for t in texts2), "轮2: 新订单查询 Flow 正常执行")

    snap2 = await app._graph.aget_state(config)
    paused_refund = snap2.values.get("paused_flows", {}).get("refund_request")
    check(paused_refund is not None, "旧退款 Flow 已进入 paused_flows")
    check(paused_refund.get("step_id") == "ask_refund_reason",
          "暂停快照保存精确 step_id=ask_refund_reason")
    check(paused_refund.get("slots", {}).get("order_number") == "A001",
          "暂停快照保留原订单号")
    check("refund_reason" not in paused_refund.get("slots", {}),
          "切换意图文本没有被误写成 refund_reason")

    r3 = await app.chat(text_msg("继续刚才的退款", sender="switch-u"))
    texts3 = [m.text for m in r3.messages]
    check(any("继续刚才" in t for t in texts3), "轮3: resume_flow(flow=refund_request) 命中指定 Flow")
    check(any("退款原因" in t for t in texts3), "轮3: 恢复到原 ask_refund_reason 而不是丢失现场")

    snap3 = await app._graph.aget_state(config)
    check(snap3.values.get("active_flow") == "refund_request", "恢复后退款 Flow 重新变为 active")
    check("refund_request" not in snap3.values.get("paused_flows", {}), "指定 Flow 已从暂停栈弹出")

    # 轮4：正常回答当前 collect；FakePlanner 自动映射到 refund_reason。
    r4 = await app.chat(text_msg("商品破损", sender="switch-u"))
    check(any("售后处理建议" in m.text for m in r4.messages), "轮4: 恢复后的 Flow 可继续走完")

    # 旧 checkpoint 兼容：历史版本 paused_flows 只有 slots，没有 step_id。
    from atguigu.graph.state import unpack_paused_flow_snapshot
    legacy_step, legacy_slots = unpack_paused_flow_snapshot({"order_number": "LEGACY-001"})
    check(legacy_step is None and legacy_slots["order_number"] == "LEGACY-001",
          "旧 slots-only paused checkpoint 仍可读取")


async def test_validator_guards():
    print("[7] Validator 安全闸门：flow / slot / intent 白名单")
    from atguigu.knowledge.intents import KNOWLEDGE_INTENTS
    from atguigu.plan.turn_plan import ClarifyReason, KnowledgeTurnPlan
    from atguigu.plan.validator import TurnPlanValidator

    flows_list = FlowLoader().load_multi_yaml([
        PROJECT_ROOT / "flow_config" / "system_flows.yml",
        PROJECT_ROOT / "flow_config" / "user_flows.yml",
    ])
    validator = TurnPlanValidator()

    # 模型不能启动系统内部 flow
    p1 = TurnPlan(task=TaskTurnPlan(commands=[
        StartFlowCommand(command="start_flow", flow="system_task_started"),
    ]))
    r1 = validator.validate(p1, None, flows_list, KNOWLEDGE_INTENTS)
    check(r1.reason is ClarifyReason.UNKNOWN_TASK_FLOW, "系统 flow 不能由模型启动")

    # order_status_query 只允许 order_number，不允许模型乱写 refund_reason
    p2 = TurnPlan(task=TaskTurnPlan(commands=[
        StartFlowCommand(command="start_flow", flow="order_status_query"),
        SetSlotsCommand(command="set_slots", slots={"refund_reason": "太贵了"}),
    ]))
    r2 = validator.validate(p2, None, flows_list, KNOWLEDGE_INTENTS)
    check(r2.reason is ClarifyReason.INVALID_TASK_SLOTS, "非法槽位 key 被拒绝")

    # 当前活跃订单流程中，正常补 order_number 应通过
    p3 = TurnPlan(task=TaskTurnPlan(commands=[
        SetSlotsCommand(command="set_slots", slots={"order_number": "A001"}),
    ]))
    r3 = validator.validate(
        p3,
        None,
        flows_list,
        KNOWLEDGE_INTENTS,
        active_flow="order_status_query",
    )
    check(r3.valid, "活跃流程允许补充白名单槽位")

    # 未知知识 intent 不再 KeyError，而是确定性拒绝
    p4 = TurnPlan(knowledge=KnowledgeTurnPlan(intents=["made_up_intent"]))
    r4 = validator.validate(p4, None, flows_list, KNOWLEDGE_INTENTS)
    check(r4.reason is ClarifyReason.UNKNOWN_KNOWLEDGE_INTENT, "未知知识 intent 被拒绝")

    # 空 intent 有明确原因码
    p5 = TurnPlan(knowledge=KnowledgeTurnPlan(intents=[]))
    r5 = validator.validate(p5, None, flows_list, KNOWLEDGE_INTENTS)
    check(r5.reason is ClarifyReason.MISSING_KNOWLEDGE_INTENT, "空知识 intent 被拒绝")

    # 未知 command 先解析成基类，再由 validator 拦截，避免请求直接 500
    p6 = TurnPlan.from_dict({
        "task": {"commands": [{"command": "delete_order", "flow": "order_status_query"}]},
        "knowledge": None,
        "chitchat": None,
    })
    r6 = validator.validate(p6, None, flows_list, KNOWLEDGE_INTENTS)
    check(r6.reason is ClarifyReason.INVALID_TASK_COMMANDS, "未知 command 进入 validator 后被拒绝")


async def test_refund_read_only_boundary():
    print("[8] 旺店通售后建议：只读边界")
    plans = [TurnPlan(task=TaskTurnPlan(commands=[
        StartFlowCommand(command="start_flow", flow="refund_request"),
        SetSlotsCommand(
            command="set_slots",
            slots={"order_number": "WD-RISK-001", "refund_reason": "商品破损"},
        ),
    ]))]
    app = build_app(make_saver(), plans)

    result = await app.chat(text_msg("订单WD-RISK-001商品破损，客户要求退款", sender="refund-u"))
    texts = [m.text for m in result.messages]
    check(any("售后处理建议" in t for t in texts), "生成售后处理建议")
    check(any("不会直接修改订单或提交退款" in t for t in texts), "明确禁止模型直接执行退款")
    check(any("风险等级" in t for t in texts), "输出风险等级供人工复核")


async def test_recommendation_fallback():
    print("[9] 相似商品推荐：中台不可用时安全降级")
    plans = [TurnPlan(task=TaskTurnPlan(commands=[
        StartFlowCommand(command="start_flow", flow="similar_product_recommendation"),
        SetSlotsCommand(command="set_slots", slots={"product_id": "SKU-NOT-FOUND"}),
    ]))]
    app = build_app(make_saver(), plans)

    result = await app.chat(text_msg("给SKU-NOT-FOUND推荐相似商品", sender="recommend-u"))
    texts = [m.text for m in result.messages]
    check(any("暂时没有查到" in t for t in texts), "推荐接口无结果时返回可解释降级")


async def test_handoff_without_agent():
    print("[10] 人工转接：无在线坐席时安全降级")
    plans = [TurnPlan(task=TaskTurnPlan(commands=[
        StartFlowCommand(command="start_flow", flow="human_handoff"),
    ]))]
    app = build_app(make_saver(), plans)

    result = await app.chat(text_msg("我要转人工", sender="handoff-u"))
    texts = [m.text for m in result.messages]
    check(any("没有在线人工客服" in t for t in texts), "无坐席时不假装转接成功")
    check(any("不会直接执行" in t for t in texts), "高风险动作继续保持人工边界")


async def test_agent_session_ownership():
    print("[11] 人工坐席：会话归属校验")
    from atguigu.api.transfer_manager import TransferManager

    class FakeConnections:
        def __init__(self):
            self.user_messages = []
            self.agent_messages = []

        async def send_to_user(self, sender_id, payload):
            self.user_messages.append((sender_id, payload))
            return True

        async def send_to_agent(self, agent_id, payload):
            self.agent_messages.append((agent_id, payload))
            return True

        async def broadcast_to_agents(self, payload):
            return None

        def has_online_agents(self):
            return True

    connections = FakeConnections()
    manager = TransferManager(connections)
    session = manager.get_or_create_session("buyer-001")
    session.mode = "human"
    session.agent_id = "agent-a"

    bad_chat = await manager.agent_chat("agent-b", "buyer-001", "越权消息")
    check(not bad_chat, "其他坐席不能向已绑定用户发送消息")
    check(not connections.user_messages, "越权消息没有下发给用户")

    good_chat = await manager.agent_chat("agent-a", "buyer-001", "正常回复")
    check(good_chat, "绑定坐席可以正常回复")

    bad_close = await manager.agent_close_session("agent-b", "buyer-001")
    check(not bad_close and session.mode == "human", "其他坐席不能结束该人工会话")

    good_close = await manager.agent_close_session("agent-a", "buyer-001")
    check(good_close and session.mode == "machine", "绑定坐席可以结束会话并切回机器人")


async def test_knowledge_and_chitchat():
    print("[12] 知识/闲聊轨道 + 无流程卡片澄清")
    plans = [
        TurnPlan(knowledge=KnowledgeTurnPlan(intents=["refund_policy"])),
        TurnPlan(chitchat=ChitChatTurnPlan(chat="你好")),
    ]
    app = build_app(make_saver(), plans)

    r1 = await app.chat(text_msg("退货政策"))
    check(r1.messages and r1.messages[0].text == "[知识:refund_policy]", "知识轨道命中")

    r2 = await app.chat(text_msg("你好"))
    check(r2.messages and r2.messages[0].text == "[闲聊:你好]", "闲聊轨道命中")

    # 无流程时点卡片 → OBJECT_REQUIRES_INTENT 澄清
    r3 = await app.chat(card_msg("A001"))
    check(any("[澄清:object_requires_intent]" in m.text for m in r3.messages), "无流程卡片→澄清")


if __name__ == "__main__":
    asyncio.run(test_topology())
    asyncio.run(test_flow_with_interrupt_collect())
    asyncio.run(test_multi_intent_pending())
    asyncio.run(test_card_direct_fill())
    asyncio.run(test_checkpointer_persistence())
    asyncio.run(test_interrupt_switch_and_precise_resume())
    asyncio.run(test_validator_guards())
    asyncio.run(test_refund_read_only_boundary())
    asyncio.run(test_recommendation_fallback())
    asyncio.run(test_handoff_without_agent())
    asyncio.run(test_agent_session_ownership())
    asyncio.run(test_knowledge_and_chitchat())
    print("\n全部通过 ✅ 旺店通 LangGraph：interrupt意图切换、精确Flow恢复、历史持久化、Validator白名单、售后只读、推荐降级、人工转接均正常")
