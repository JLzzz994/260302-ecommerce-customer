"""
图状态（Graph State）：全部业务状态都在这里，由 checkpointer 持久化

通道说明：
- messages：add_messages reducer，对话历史（HumanMessage/AIMessage），卡片放 additional_kwargs
- flow_step：当前流程步骤指针（"flow_id:step_id"），是 advance_flow 自循环推进的依据
- user_message / turn_plan / validated：单轮中间产物
"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from atguigu.domain.messages import UserMessage
from atguigu.plan.turn_plan import TurnPlan, TurnPlanValidatedResult


class PausedFlowSnapshot(TypedDict, total=False):
    """业务级暂停快照。

    新版保存 step_id + slots，支持精确恢复；读取时仍兼容旧 checkpoint
    的 flow_id -> slots 结构。
    """

    step_id: str | None
    slots: dict[str, Any]


class GraphState(TypedDict, total=False):
    # ===== 身份 =====
    sender_id: str

    # ===== 对话历史（add_messages：追加合并，自动持久化） =====
    messages: Annotated[list[AnyMessage], add_messages]

    # ===== 本轮输入与中间产物（不跨轮） =====
    user_message: UserMessage
    turn_plan: TurnPlan
    validated: TurnPlanValidatedResult

    # ===== 流程运行时（跨轮，checkpoint 持久化） =====
    slots: dict[str, Any]                  # 当前活跃流程的槽位
    active_flow: str | None                # 当前流程ID（None=无流程进行中）
    flow_step: str | None                  # 当前执行步骤 "flow_id:step_id"
    resume_step: str | None                # 重新进入子图时的精确入口 step_id；首个节点消费后清空
    interrupt_handoff: bool                # collect 恢复输入被判定为新意图，子图退出后交回父图
    paused_flows: dict[str, dict[str, Any]] # flow_id -> {step_id, slots}；兼容旧 slots-only 快照
    flow_context: dict[str, Any] | None    # 过场话术渲染变量（started_flow_name 等）

    # ===== 卡片 =====
    focused_object: dict[str, Any] | None  # 用户最近点击的订单/商品卡片

    # ===== 挂起意图队列（多意图被拒时暂存，随 checkpoint 持久化） =====
    pending_intents: list[dict[str, Any]]  # [{"flows": [...], "knowledge_intents": [...]}]
    last_completed_flow: str | None        # 一次性完成标记（end步骤设置）


def flow_step_key(flow_id: str, step_id: str) -> str:
    return f"{flow_id}:{step_id}"


def parse_flow_step(key: str | None) -> tuple[str, str] | None:
    if not key:
        return None
    flow_id, _, step_id = key.partition(":")
    return flow_id, step_id


def build_paused_flow_snapshot(step_id: str | None, slots: dict[str, Any]) -> PausedFlowSnapshot:
    """创建新版暂停快照，复制 slots，避免后续活跃流程修改污染快照。"""
    return PausedFlowSnapshot(step_id=step_id, slots=dict(slots))


def unpack_paused_flow_snapshot(snapshot: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """读取暂停快照，并兼容旧版 flow_id -> slots 的 checkpoint。"""
    if "slots" in snapshot and ("step_id" in snapshot or len(snapshot) <= 2):
        return snapshot.get("step_id"), dict(snapshot.get("slots") or {})
    return None, dict(snapshot)
