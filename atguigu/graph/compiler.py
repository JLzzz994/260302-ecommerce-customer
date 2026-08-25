"""
流程编译器：把 YAML 里的一个业务流程（Flow）编译成一张 LangGraph 子图

编译规则（对照原 FlowExecutor 的运行时解释执行）：
- start 步骤：直接跳到 next 指向的步骤（编译期为纯边，无节点）
- action 步骤：一个节点——运行 Action，把消息追加进 messages、把返回 slots 合并进 slots，
  然后按 next（静态边/条件边）路由到下一个节点
- collect 步骤：一个节点——槽位没值时用 interrupt() 暂停整图等用户回答
  （渲染 YAML 里的提问话术发给用户），恢复时把回答写进 slots 再校验；
  校验不过则删掉槽位重新 interrupt，与原 system_collect_information 两次进入语义一致
- end 步骤：标记流程完成（active_flow=None + last_completed_flow），落到 END

与原实现的本质区别：原 FlowExecutor 是【运行时解释器】（while true 沿 steps 推进指针），
这里把推进编译成【静态图拓扑】——节点间的边在启动时确定，运行时不再查表跳转；
唯一保留的"指针"是条件边的选择函数（slots 条件评估），语义与原 _eval_condition 一致。
"""

import re
from typing import Any, Callable

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from atguigu.domain.messages import BotMessage
from atguigu.graph.context import TurnContext
from atguigu.graph.state import GraphState
from atguigu.task.action.runner import ActionRunner, ActionCall
from atguigu.task.flows.flows import Flow
from atguigu.task.flows.steps import (FlowStep, ActionFlowStep, CollectFlowStep, EndFlowStep,
                                      FlowStepStaticLink, FlowStepConditionLink, FlowStepFallbackLink)
from atguigu.task.flows.links import FlowStepLink


def make_turn_context(gs: GraphState) -> TurnContext:
    """从图状态构建组件读视图（本轮消息 + 排除本轮输入的历史）"""
    history = [m for m in gs.get("messages", []) if not getattr(m, "additional_kwargs", {}).get("_this_turn")]
    return TurnContext(
        user_message=gs.get("user_message"),
        history_messages=history,
        slots=gs.get("slots") or {},
        flow_context=gs.get("flow_context"),
        focused_object=gs.get("focused_object"),
    )


def eval_condition(condition: str, slots: dict[str, Any]) -> bool:
    """条件边/槽位校验的条件评估（与原 executor._eval_condition 相同的可控上下文）"""
    return bool(eval(condition, {}, {"slots": slots}))  # noqa: S307 条件来自自家YAML配置


def _first_static_next(step: FlowStep) -> str | None:
    for link in step.next:
        if isinstance(link, FlowStepStaticLink):
            return link.target
    return None


class FlowCompiler:

    def __init__(self, action_runner: ActionRunner):
        self._action_runner = action_runner

    def compile_flow(self, flow: Flow) -> Any:
        """把一个流程编译成子图（返回 compiled subgraph，可被 add_node 挂进主图）"""
        builder = StateGraph(GraphState)

        steps = {step.id: step for step in flow.steps}
        entry = self._entry_step_id(flow)

        # START → 入口步骤（start步骤是纯跳板：编译期把它的出边接到START上，
        # 含条件出边的 start（如 similar_product_recommendation）用条件边路由）
        first_step = steps[entry]
        if self._is_passthrough(first_step):
            if all(isinstance(link, FlowStepStaticLink) for link in first_step.next):
                builder.add_edge(START, f"step_{first_step.next[0].target}")
            else:
                self._add_step_edges(builder, first_step, from_start=True)
        else:
            builder.add_edge(START, f"step_{entry}")

        # 每个步骤注册为节点 + 出边（start 跳板被绕过；end 是 terminal 节点）
        for step in flow.steps:
            if self._is_passthrough(step):
                continue

            builder.add_node(f"step_{step.id}", self._make_step_node(flow, step))

            if step.__class__.__name__ == "EndFlowStep":
                builder.add_edge(f"step_{step.id}", END)  # end：打完成标记后直接结束子图
            elif isinstance(step, (ActionFlowStep, CollectFlowStep)):
                if not step.next:
                    builder.add_edge(f"step_{step.id}", END)  # 防御：无出边直达END
                else:
                    self._add_step_edges(builder, step)
            else:
                builder.add_edge(f"step_{step.id}", END)

        return builder.compile()

    # ============================== 步骤节点工厂 ==============================

    def _make_step_node(self, flow: Flow, step: FlowStep) -> Callable:
        if isinstance(step, ActionFlowStep):
            return self._make_action_node(flow, step)
        if isinstance(step, CollectFlowStep):
            return self._make_collect_node(flow, step)
        return self._make_terminal_node(flow)

    def _make_action_node(self, flow: Flow, step: ActionFlowStep) -> Callable:
        """action 步骤：执行 Action（说/做），合并消息与槽位"""
        action_runner = self._action_runner

        async def action_node(gs: GraphState) -> dict[str, Any]:
            ctx = make_turn_context(gs)

            # system_collect_information 里 args 为字符串 "context.response" 的特殊分支不再存在：
            # LangGraph 版收集提问直接由 collect 节点渲染，不经过 action
            action_kwargs = step.args if not isinstance(step.args, str) else {}

            result = await action_runner.run(ActionCall(action_name=step.action, action_kwargs=action_kwargs), ctx)

            updates: dict[str, Any] = {"flow_step": f"{flow.flow_id}:{step.id}"}

            # 1. 消息（action_response 产生）
            if result.messages:
                updates["messages"] = [AIMessage(content=m.text, additional_kwargs={"object": m.object_to_dict() if hasattr(m, "object_to_dict") else None})
                                       for m in result.messages]
            # 2. 槽位（action_xxx 产生）
            if result.slots:
                merged = dict(gs.get("slots") or {})
                merged.update(result.slots)
                updates["slots"] = merged

            return updates

        return action_node

    def _make_collect_node(self, flow: Flow, step: CollectFlowStep) -> Callable:
        """
        collect 步骤：LangGraph interrupt 的标准用法
        1. 槽位已有值 → （可选校验）通过则放行；不通过删值重问
        2. 槽位没有值 → 提问话术进 messages + interrupt() 暂停整图；
           下一轮用户输入作为 Command(resume=...) 恢复本节点，answer 即用户回答
        3. 卡片直填：用户点击的订单/商品卡片正好是本步骤要的槽位 → 免提问直接填
        """
        slot_name = step.slot_name

        async def collect_node(gs: GraphState) -> dict[str, Any]:
            from langchain_core.messages import AIMessage
            slots = dict(gs.get("slots") or {})
            updates: dict[str, Any] = {"flow_step": f"{flow.flow_id}:{step.id}"}

            # 0. 卡片直填（对象消息路径：用户点了订单/商品卡片）
            card_slot = self._card_fill_slot(gs, slot_name, slots)
            if card_slot is not None:
                slots.update(card_slot)

            # 1. 还缺值 → interrupt 暂停（提问文案放 payload；
            #    注意 interrupt 抛异常时节点的部分更新不会被提交，
            #    所以提问消息由 DialogueApp 检测到中断后从 payload 补发）
            if not slots.get(slot_name):
                question = self._render_question(step.response.text if isinstance(step.response.text, str) else "",
                                                 slots, gs)
                answer = interrupt({"question": question, "slot": slot_name})

                # 恢复点：answer 是本轮用户输入（文本 or 卡片ID）
                value = self._extract_answer_value(answer)
                if value:
                    slots[slot_name] = value
                    return {**updates, "slots": slots}
                else:
                    # 用户这条消息没回答（比如又点了张无关卡片）——重新中断继续等
                    return self._re_interrupt(step, slots, gs, updates)

            # 2. 已有值 → 校验（配置了才校验）
            if step.validate is not None:
                if eval_condition(step.validate.condition, slots):
                    return {**updates, "slots": slots}  # 放行，走 next 边
                # 校验不过：删除槽位重新问（与原实现语义一致）
                slots.pop(slot_name, None)
                failure_text = step.validate.failure_response.text if step.validate.failure_response else None
                question = self._render_question(failure_text, slots, gs) if failure_text else "您填写的信息有误，请重新填写。"
                answer = interrupt({"question": question, "slot": slot_name})
                value = self._extract_answer_value(answer)
                if value and eval_condition(step.validate.condition, {**slots, slot_name: value}):
                    slots[slot_name] = value
                    return {**updates, "slots": slots}
                # 还是不行：重新中断（下轮继续问）
                return self._re_interrupt(step, slots, gs, updates)

            return {**updates, "slots": slots}

        return collect_node

    def _make_terminal_node(self, flow: Flow) -> Callable:
        """流程走到 end：清空活跃流程 + 打完成标记（供主图追问挂起意图）"""
        async def end_node(gs: GraphState) -> dict[str, Any]:
            return {"active_flow": None,
                    "flow_step": None,
                    "flow_context": None,
                    "last_completed_flow": flow.flow_id}
        return end_node

    # ============================== 边 ==============================

    def _add_step_edges(self, builder: StateGraph, step: FlowStep, from_start: bool = False) -> None:
        src = START if from_start else f"step_{step.id}"

        # 纯静态边：一条直达
        if all(isinstance(link, FlowStepStaticLink) for link in step.next):
            builder.add_edge(src, self._target_of(step.next[0]))
            return

        # 带条件/兜底边：路由函数闭包捕获链接定义
        links = list(step.next)
        static_target = None
        for link in links:
            if isinstance(link, FlowStepStaticLink):
                static_target = link.target

        def route(gs: GraphState) -> str:
            slots = gs.get("slots") or {}
            fallback = static_target
            for link in links:
                if isinstance(link, FlowStepConditionLink) and eval_condition(link.condition, slots):
                    return self._target_of(link)
                if isinstance(link, FlowStepFallbackLink):
                    fallback = link.target
            return self._target_of_raw(fallback)

        builder.add_conditional_edges(src, route)

    def _target_of(self, link: FlowStepLink) -> str:
        return self._target_of_raw(link.target)

    def _target_of_raw(self, target: str) -> str:
        return target if target == "__end__" else f"step_{target}"

    # ============================== 工具 ==============================

    def _entry_step_id(self, flow: Flow) -> str:
        for step in flow.steps:
            if step.type.value == "start" or step.__class__.__name__ == "StartFlowStep":
                return step.id
        return flow.steps[0].id

    def _is_passthrough(self, step: FlowStep) -> bool:
        """start 是纯跳板（无副作用），end 是 terminal 节点单独处理"""
        return step.__class__.__name__ == "StartFlowStep"

    def _required_static_next(self, step: FlowStep) -> str:
        target = _first_static_next(step)
        if target is None:
            raise ValueError(f"流程步骤 {step.id} 缺少 next 指向（YAML 配置错误）")
        return target

    def _card_fill_slot(self, gs: GraphState, slot_name: str, slots: dict) -> dict | None:
        """用户点击卡片可直接填对应槽位（与原 _try_fill_slots_from_focused_object 一致）"""
        obj = gs.get("focused_object")
        if not obj:
            return None
        mapping = {"order": "order_number", "product": "product_id"}
        expected = mapping.get(obj.get("type"))
        if expected == slot_name and not slots.get(slot_name):
            return {slot_name: obj.get("id")}
        return None

    def _render_question(self, text: str, slots: dict, gs: GraphState) -> str:
        from jinja2 import Template
        return Template(text).render(slots=slots, context=gs.get("flow_context") or {})

    def _extract_answer_value(self, answer: Any) -> Any:
        """从恢复值里取槽位值：直接值 / {"text":..} / 卡片 {"object_id":..}"""
        if answer is None:
            return None
        if isinstance(answer, str):
            return answer.strip() or None
        if isinstance(answer, dict):
            if answer.get("object_id"):
                return answer["object_id"]
            text = answer.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
        return answer

    def _re_interrupt(self, step: CollectFlowStep, slots: dict, gs: GraphState, updates: dict) -> dict:
        """继续等待用户回答：再次 interrupt（本轮已发过提问，不再重复发消息）"""
        interrupt({"question": "", "slot": step.slot_name, "silent": True})
        return {**updates, "slots": slots}
