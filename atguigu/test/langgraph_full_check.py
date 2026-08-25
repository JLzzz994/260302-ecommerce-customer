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
from atguigu.plan.commands import StartFlowCommand, SetSlotsCommand
from atguigu.task.flows.loader import FlowLoader
from atguigu.task.action.buidler import build_action_runner
from atguigu.graph.main_graph import build_main_graph
from atguigu.graph.app import DialogueApp
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def check(cond: bool, tip: str):
    assert cond, f"FAIL: {tip}"
    print(f"  ✓ {tip}")


def text_msg(text: str, sender="u1") -> UserMessage:
    return UserMessage(sender_id=sender, message_id="m", type=MessageType.TEXT, text=text)


def card_msg(obj_id="A001", obj_type="order", sender="u1") -> UserMessage:
    return UserMessage(sender_id=sender, message_id="m", type=MessageType.OBJECT,
                       object=FocusedObject(id=obj_id, type=obj_type, title=obj_id, attributes={}))


# ============================== 假组件 ==============================

class FakePlanner:
    def __init__(self, plans: list[TurnPlan]):
        self._plans = list(plans)

    async def predict(self, ctx, flows_list, knowledge_intents, **kwargs) -> TurnPlan:
        return self._plans.pop(0)


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
    for expected in ("ingest", "plan", "validate", "task", "knowledge", "chitchat",
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
    check(any("请告诉我你的订单号" in t for t in texts1), "轮1: collect 中断并发出提问")

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
    check(any("退款申请" in t and "继续帮你处理" in t for t in texts2), "轮2: 流程end后追问挂起的退款意图")
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
    app2 = DialogueApp(*[app1._graph, saver])  # 复用图；真实场景是重新 build_main_graph
    from atguigu.graph.app import DialogueApp as DA
    app2 = build_app(saver, [])  # planner空脚本：恢复路径不再调planner
    r = await app2.chat(text_msg("A999", sender="u9"))
    texts = [m.text for m in r.messages]
    check(any("订单A999当前状态" in t for t in texts), "重启后从中断点恢复，用户回答直达流程")


async def test_knowledge_and_chitchat():
    print("[6] 知识/闲聊轨道 + 无流程卡片澄清")
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
    asyncio.run(test_knowledge_and_chitchat())
    print("\n全部通过 ✅ LangGraph 全量改造：interrupt收集/恢复、多意图挂起、卡片、持久化均正常")
