"""
主图装配：整个对话系统的 LangGraph 图

主图拓扑（每轮用户消息进入，从 START 跑到 END 或 interrupt 暂停等用户）：

    START → ingest ─┬─(有活跃流程)→ flow_<flow_id> 子图 ──→ pending_follow_up → END
                    │                    ▲  (collect 节点 interrupt 等槽位，恢复即继续)
                    ├─(无流程)→ plan → validate ─┬→ task →（设置指针）→ flow 子图
                    │                           ├→ knowledge → END
                    │                           ├→ chitchat  → END
                    │                           └→ clarify   → END
                    └─(卡片+无流程)→ object_dispatch → END

设计要点：
1. 每个业务流程（YAML）被 FlowCompiler 编译成子图，作为节点挂进主图；
   "执行哪个流程"由条件边按 active_flow 指针分发——图上的动态分发。
2. 流程中途用户新消息 → ingest 条件边发现 active_flow 有值 → 直接路由回流程子图，
   子图从中断点恢复（Command(resume=用户输入) 由 DialogueApp 注入）——
   这就是"多轮槽位收集"的 LangGraph 原生实现，替代原 system_collect_information 过场。
3. 任务轨道的 task 节点 = 原命令处理器（start/resume/cancel/set_slots）的状态迁移，
   产物是"指针+槽位"，随后条件边把控制交给对应流程子图。
4. 挂起意图队列（多意图被拒时压栈、流程完成后追问）语义与旧引擎完全一致，状态在图状态里随 checkpoint 持久化。
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from atguigu.domain.messages import UserMessage, BotMessage, MessageType
from atguigu.graph.compiler import FlowCompiler, make_turn_context
from atguigu.graph.state import (
    GraphState,
    build_paused_flow_snapshot,
    parse_flow_step,
    unpack_paused_flow_snapshot,
)
from atguigu.plan.turn_plan import ClarifyReason


def build_main_graph(*,
                     planner,
                     validator,
                     clarify_responder,
                     knowledge_handler,
                     chitchat_handler,
                     flows_list,
                     action_runner,
                     checkpointer: BaseCheckpointSaver | None = None):
    knowledge_intents = knowledge_handler.knowledge_intents

    async def route_interrupted_turn(
        gs: GraphState,
        flow_id: str,
        step_id: str,
        slot_name: str,
        answer: Any,
    ) -> dict[str, Any]:
        """collect 恢复输入先过 Planner：槽位答案原地继续，新意图交回父图。"""
        ctx = make_turn_context(gs)
        turn_plan = await planner.predict(
            ctx,
            flows_list,
            knowledge_intents,
            active_flow=flow_id,
            active_flow_step=step_id,
            slots=gs.get("slots") or {},
            paused_flows=gs.get("paused_flows") or {},
        )
        validated = validator.validate(
            turn_plan,
            gs.get("focused_object"),
            flows_list,
            knowledge_intents,
            active_flow=flow_id,
        )

        # Planner/Validator 不确定时保持原 collect 语义，避免把正常槽位答案误拦截。
        if not validated.valid:
            return {"kind": "answer"}

        if turn_plan.task is not None:
            commands = turn_plan.task.commands
            has_control = any(
                command.command in {"start_flow", "resume_flow", "cancel_flow"}
                for command in commands
            )
            if not has_control:
                for command in commands:
                    if command.command == "set_slots" and slot_name in command.slots:
                        return {"kind": "slot", "value": command.slots[slot_name]}
                return {"kind": "answer"}

        # task 控制命令，或 knowledge/chitchat：先结束当前子图 invocation，
        # 再由父图 interrupt_dispatch 做真正的暂停/切换。
        return {
            "kind": "parent",
            "turn_plan": turn_plan,
            "validated": validated,
        }

    compiler = FlowCompiler(
        action_runner,
        interrupted_turn_router=route_interrupted_turn,
    )

    # 每个业务流程编译成子图，节点名 flow_<flow_id>
    flow_subgraphs = {flow.flow_id: compiler.compile_flow(flow) for flow in flows_list.flows
                      if not flow.flow_id.startswith("system_")}
    flow_node_names = [f"flow_{fid}" for fid in flow_subgraphs]

    # ============================== 节点 ==============================

    async def ingest(gs: GraphState) -> dict[str, Any]:
        """
        入口节点：本轮用户消息进 messages 通道（带 _this_turn 标记，构建 TurnContext
        历史时排除自己）；卡片消息进 focused_object
        """
        user_message: UserMessage = gs["user_message"]
        updates: dict[str, Any] = {"sender_id": user_message.sender_id}

        if user_message.type is MessageType.OBJECT and user_message.object is not None:
            updates["focused_object"] = user_message.object.to_dict()
            updates["messages"] = [HumanMessage(
                content=(user_message.text or user_message.object.title or ""),
                additional_kwargs={"object": user_message.object.to_dict(), "_turn_message_id": user_message.message_id},
            )]
        else:
            updates["messages"] = [HumanMessage(
                content=user_message.text or "",
                additional_kwargs={"_turn_message_id": user_message.message_id},
            )]
        return updates

    async def plan(gs: GraphState) -> dict[str, Any]:
        ctx = make_turn_context(gs)
        step_id = (parse_flow_step(gs.get("flow_step")) or (None, None))[1]
        turn_plan = await planner.predict(
            ctx, flows_list, knowledge_intents,
            active_flow=gs.get("active_flow"),
            active_flow_step=step_id,
            slots=gs.get("slots") or {},
            paused_flows=gs.get("paused_flows") or {},
        )
        return {"turn_plan": turn_plan}

    async def validate(gs: GraphState) -> dict[str, Any]:
        turn_plan = gs["turn_plan"]
        validated = validator.validate(
            turn_plan,
            gs.get("focused_object"),
            flows_list,
            knowledge_intents,
            active_flow=gs.get("active_flow"),
        )

        updates: dict[str, Any] = {"validated": validated}

        # 多意图被拒 → 压入挂起意图队列（与旧引擎钩子同语义）
        if not validated.valid and validated.reason in (ClarifyReason.MULTIPLE_TRACKS,
                                                        ClarifyReason.MULTIPLE_TASK_FLOWS):
            pending = list(gs.get("pending_intents") or [])
            flows = [c.flow for c in (turn_plan.task.commands if turn_plan.task else [])
                     if c.__class__.__name__ == "StartFlowCommand" and getattr(c, "flow", None)]
            intents = list(turn_plan.knowledge.intents) if turn_plan.knowledge is not None else []
            if flows or intents:
                entry = {"flows": flows, "knowledge_intents": intents}
                if entry not in pending:
                    pending.append(entry)
                    del pending[:-3]
                    updates["pending_intents"] = pending

        return updates

    def _current_step_id(gs: GraphState, flow_id: str | None) -> str | None:
        parsed = parse_flow_step(gs.get("flow_step"))
        if parsed and flow_id and parsed[0] == flow_id:
            return parsed[1]
        return None

    async def task(gs: GraphState) -> dict[str, Any]:
        """处理 start/resume/cancel/set_slots，并维护可精确恢复的业务暂停栈。"""
        turn_plan = gs["turn_plan"]
        commands = turn_plan.task.commands

        slots = dict(gs.get("slots") or {})
        active_flow = gs.get("active_flow")
        paused = dict(gs.get("paused_flows") or {})
        flow_context = gs.get("flow_context")
        resume_step = gs.get("resume_step")
        next_flow_step = gs.get("flow_step")
        pending = [dict(e) for e in gs.get("pending_intents") or []]
        intro_messages: list[AIMessage] = []
        cancel_message = None

        def remove_flow(entry: dict, flow_id: str) -> dict:
            return {
                "flows": [f for f in entry.get("flows", []) if f != flow_id],
                "knowledge_intents": entry.get("knowledge_intents", []),
            }

        def pause_current() -> None:
            nonlocal paused
            if not active_flow:
                return
            paused[active_flow] = build_paused_flow_snapshot(
                _current_step_id(gs, active_flow),
                slots,
            )

        for command in commands:
            cname = command.command

            if cname == "start_flow":
                target_flow = command.flow
                target = flows_list.get_flow_by_flow_id(target_flow)

                if active_flow and active_flow != target_flow:
                    pause_current()
                    current = flows_list.get_flow_by_flow_id(active_flow)
                    flow_context = {
                        "interrupted_flow_name": current.flow_name,
                        "started_flow_name": target.flow_name,
                    }
                    intro_messages.append(
                        AIMessage(content=f"好的，我们先把{current.flow_name}放一放。")
                    )
                elif active_flow is None:
                    flow_context = {"started_flow_name": target.flow_name}

                # start 是“重新开始”语义：同名旧暂停快照失效。
                paused.pop(target_flow, None)
                intro_messages.append(AIMessage(content=f"好的，我们先处理{target.flow_name}。"))
                active_flow = target_flow
                slots = {}
                resume_step = None
                next_flow_step = f"{active_flow}:start"
                pending = [remove_flow(e, target_flow) for e in pending]

            elif cname == "resume_flow":
                requested = getattr(command, "flow", None)
                target_flow = None

                if requested and requested in paused:
                    target_flow = requested
                elif requested is None and paused:
                    target_flow = next(reversed(paused))

                if target_flow is not None:
                    if active_flow and active_flow != target_flow:
                        pause_current()

                    step_id, restored_slots = unpack_paused_flow_snapshot(
                        paused.pop(target_flow)
                    )
                    active_flow = target_flow
                    slots = restored_slots
                    resume_step = step_id
                    next_flow_step = f"{target_flow}:{step_id or 'start'}"
                    flow_context = {
                        "resumed_flow_name": flows_list.get_flow_by_flow_id(
                            target_flow
                        ).flow_name
                    }
                    intro_messages.append(
                        AIMessage(
                            content=f"好的，我们继续刚才的"
                            f"{flows_list.get_flow_by_flow_id(target_flow).flow_name}。"
                        )
                    )

            elif cname == "cancel_flow":
                if active_flow:
                    canceled_name = flows_list.get_flow_by_flow_id(active_flow).flow_name
                    cancel_message = AIMessage(content=f"好的，已取消{canceled_name}。")
                    active_flow = None
                    slots = {}
                    flow_context = None
                    resume_step = None
                    next_flow_step = None
                    # 只取消当前 Flow，不清空其他 paused_flows，允许随后精确恢复旧任务。

            elif cname == "set_slots":
                slots.update(command.slots)

        pending = [
            e for e in pending
            if e.get("flows") or e.get("knowledge_intents")
        ]

        updates = {
            "active_flow": active_flow,
            "flow_step": next_flow_step if active_flow else None,
            "resume_step": resume_step if active_flow else None,
            "slots": slots,
            "paused_flows": paused,
            "flow_context": flow_context if active_flow else None,
            "pending_intents": pending,
            "interrupt_handoff": False,
        }

        if cancel_message is not None:
            return {**updates, "messages": [cancel_message]}

        return {**updates, "messages": intro_messages}

    async def interrupt_dispatch(gs: GraphState) -> dict[str, Any]:
        """collect 中途新意图退出子图后，在父图统一处理业务暂停。"""
        updates: dict[str, Any] = {"interrupt_handoff": False}
        validated = gs.get("validated")
        turn_plan = gs.get("turn_plan")

        if validated is None or not validated.valid or turn_plan is None:
            return updates

        # Task 轨道交给 task()；它需要看到当前 active_flow 才能完成 swap/pause。
        if turn_plan.task is not None:
            return updates

        # Knowledge/Chitchat 是临时 detour：先完整保存当前 Flow 的 step + slots。
        active_flow = gs.get("active_flow")
        if active_flow:
            paused = dict(gs.get("paused_flows") or {})
            paused[active_flow] = build_paused_flow_snapshot(
                _current_step_id(gs, active_flow),
                dict(gs.get("slots") or {}),
            )
            updates.update({
                "paused_flows": paused,
                "active_flow": None,
                "flow_step": None,
                "resume_step": None,
                "slots": {},
                "flow_context": None,
            })

        return updates

    async def pending_follow_up(gs: GraphState) -> dict[str, Any]:
        """
        挂起意图追问（与旧引擎 _append_pending_intents_follow_up 同语义）：
        业务流程走到 end（last_completed_flow 被打标）后弹出全部挂起意图，模板追问。
        一次性：弹出即清空。
        """
        updates: dict[str, Any] = {}
        if not gs.get("last_completed_flow"):
            return updates
        updates["last_completed_flow"] = None  # 一次性消费
        pending = list(gs.get("pending_intents") or [])
        if not pending:
            return updates
        updates["pending_intents"] = []
        labels = _pending_labels(pending)
        if labels:
            text = f"对了，你刚才还提到了：{'、'.join(labels)}。需要现在继续帮你处理吗？"
            return {**updates, "messages": [AIMessage(content=text)]}
        return updates

    def _pending_labels(entries: list[dict]) -> list[str]:
        labels = []
        for entry in entries:
            for flow_id in entry.get("flows", []):
                flow = flows_list.get_flow_by_flow_id(flow_id)
                labels.append(flow.flow_name if flow else flow_id)
            for intent_id in entry.get("knowledge_intents", []):
                meta = knowledge_intents.get(intent_id)
                labels.append(meta.description if meta else intent_id)
        return list(dict.fromkeys(labels))

    async def knowledge(gs: GraphState) -> dict[str, Any]:
        ctx = make_turn_context(gs)
        bot_messages: list[BotMessage] = await knowledge_handler.handle(ctx, gs["turn_plan"].knowledge.intents)
        # 知识问答一轮完成：直接弹出挂起意图追问（拼在回答后）
        pending = list(gs.get("pending_intents") or [])
        messages = [AIMessage(content=m.text, additional_kwargs={"object": m.object.to_dict() if m.object else None})
                    for m in bot_messages]
        if pending:
            labels = _pending_labels(pending)
            if labels:
                extra = f"\n对了，你刚才还提到了：{'、'.join(labels)}。需要现在继续帮你处理吗？"
                messages[-1] = AIMessage(content=bot_messages[-1].text + extra,
                                         additional_kwargs=messages[-1].additional_kwargs)
                pending = []
        return {"messages": messages, "pending_intents": pending}

    async def chitchat(gs: GraphState) -> dict[str, Any]:
        ctx = make_turn_context(gs)
        bot_messages = await chitchat_handler.handle(gs["turn_plan"].chitchat.chat, ctx)
        return {"messages": [AIMessage(content=m.text) for m in bot_messages]}

    async def clarify(gs: GraphState) -> dict[str, Any]:
        ctx = make_turn_context(gs)
        bot_messages = await clarify_responder.respond(gs["validated"].reason, ctx)
        return {"messages": [AIMessage(content=m.text) for m in bot_messages]}

    async def object_dispatch(gs: GraphState) -> dict[str, Any]:
        """卡片消息但无活跃流程：澄清 OBJECT_REQUIRES_INTENT（有流程时 ingest 已直送子图）"""
        ctx = make_turn_context(gs)
        bot_messages = await clarify_responder.respond(ClarifyReason.OBJECT_REQUIRES_INTENT, ctx)
        return {"messages": [AIMessage(content=m.text) for m in bot_messages]}

    # ============================== 条件边 ==============================

    def route_to_flow(gs: GraphState) -> str:
        active = gs.get("active_flow")
        return f"flow_{active}" if active in flow_subgraphs else "pending_follow_up"

    def route_after_ingest(gs: GraphState) -> str:
        user_message: UserMessage = gs["user_message"]
        if gs.get("active_flow"):
            # 有活跃流程（含 interrupt 中断点）：直送流程子图，
            # 用户输入由 DialogueApp 以 Command(resume=...) 注入恢复
            return route_to_flow(gs)
        if user_message.type is MessageType.OBJECT:
            return "object_dispatch"
        return "plan"

    def route_after_validate(gs: GraphState) -> str:
        validated = gs.get("validated")
        if validated is None or not validated.valid:
            return "clarify"
        turn_plan = gs["turn_plan"]
        if turn_plan.task is not None:
            return "task"
        if turn_plan.knowledge is not None:
            return "knowledge"
        return "chitchat"

    def route_after_task(gs: GraphState) -> str:
        # cancel 已发消息无活跃流程 → 直接追问检查
        return route_to_flow(gs) if gs.get("active_flow") else "pending_follow_up"

    def route_after_interrupt_dispatch(gs: GraphState) -> str:
        validated = gs.get("validated")
        if validated is None or not validated.valid:
            return "clarify"
        turn_plan = gs["turn_plan"]
        if turn_plan.task is not None:
            return "task"
        if turn_plan.knowledge is not None:
            return "knowledge"
        return "chitchat"

    def route_after_flow_exit(gs: GraphState) -> str:
        return "interrupt_dispatch" if gs.get("interrupt_handoff") else "pending_follow_up"

    # ============================== 装配 ==============================

    graph = StateGraph(GraphState)

    graph.add_node("ingest", ingest)
    graph.add_node("plan", plan)
    graph.add_node("validate", validate)
    graph.add_node("task", task)
    graph.add_node("interrupt_dispatch", interrupt_dispatch)
    graph.add_node("flow_exit", lambda gs: {})
    graph.add_node("pending_follow_up", pending_follow_up)
    graph.add_node("knowledge", knowledge)
    graph.add_node("chitchat", chitchat)
    graph.add_node("clarify", clarify)
    graph.add_node("object_dispatch", object_dispatch)

    # 业务流程子图作为节点挂进主图（父子图状态共享，interrupt 自动传播+持久化）
    for flow_id, subgraph in flow_subgraphs.items():
        graph.add_node(f"flow_{flow_id}", subgraph)

    graph.add_edge(START, "ingest")
    graph.add_conditional_edges("ingest", route_after_ingest,
                                ["plan", "object_dispatch"] + flow_node_names)
    graph.add_edge("plan", "validate")
    graph.add_conditional_edges("validate", route_after_validate,
                                ["task", "knowledge", "chitchat", "clarify"])
    graph.add_conditional_edges("task", route_after_task,
                                ["pending_follow_up"] + flow_node_names)
    for name in flow_node_names:
        graph.add_edge(name, "flow_exit")
    graph.add_conditional_edges(
        "flow_exit",
        route_after_flow_exit,
        ["interrupt_dispatch", "pending_follow_up"],
    )
    graph.add_conditional_edges(
        "interrupt_dispatch",
        route_after_interrupt_dispatch,
        ["task", "knowledge", "chitchat", "clarify"],
    )
    graph.add_edge("pending_follow_up", END)
    graph.add_edge("knowledge", END)
    graph.add_edge("chitchat", END)
    graph.add_edge("clarify", END)
    graph.add_edge("object_dispatch", END)

    return graph.compile(checkpointer=checkpointer)
