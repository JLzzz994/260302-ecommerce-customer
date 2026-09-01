from typing import Any

from atguigu.knowledge.intents import KnowledgeIntent
from atguigu.plan.commands import (
    CancelFlowCommand,
    ResumeFlowCommand,
    SetSlotsCommand,
    StartFlowCommand,
)
from atguigu.plan.turn_plan import (
    ClarifyReason,
    KnowledgeTurnPlan,
    TaskTurnPlan,
    TurnPlan,
    TurnPlanValidatedResult,
)
from atguigu.task.flows.flows import Flow, FlowsList


class TurnPlanValidator:
    """TurnPlan 的确定性安全闸门。

    LLM 负责理解和规划；是否允许进入业务执行层由这里裁决。
    """

    def validate(
        self,
        turn_plan: TurnPlan,
        focused_object: dict[str, Any] | None,
        flows_list: FlowsList,
        knowledge_intents: dict[str, KnowledgeIntent],
        *,
        active_flow: str | None = None,
    ) -> TurnPlanValidatedResult:
        selected_tracks = turn_plan.activated_tracks()

        if not selected_tracks:
            return self._reject(ClarifyReason.MISSING_TRACK)
        if len(selected_tracks) > 1:
            return self._reject(ClarifyReason.MULTIPLE_TRACKS)

        selected_track = selected_tracks[0]
        if selected_track == "task":
            return self._validate_task_track(
                turn_plan.task,
                flows_list,
                active_flow=active_flow,
            )
        if selected_track == "knowledge":
            return self._validate_knowledge_track(
                turn_plan.knowledge,
                focused_object,
                knowledge_intents,
            )
        return TurnPlanValidatedResult(valid=True)

    @staticmethod
    def _reject(reason: ClarifyReason) -> TurnPlanValidatedResult:
        return TurnPlanValidatedResult(valid=False, reason=reason)

    @staticmethod
    def _user_flow(flows_list: FlowsList, flow_id: str | None) -> Flow | None:
        if not flow_id or flow_id.startswith("system_"):
            return None
        return flows_list.get_flow_by_flow_id(flow_id)

    def _validate_task_track(
        self,
        task: TaskTurnPlan,
        flows_list: FlowsList,
        *,
        active_flow: str | None,
    ) -> TurnPlanValidatedResult:
        if not task.commands:
            return self._reject(ClarifyReason.MISSING_TASK_COMMANDS)

        allowed_commands = (
            StartFlowCommand,
            ResumeFlowCommand,
            CancelFlowCommand,
            SetSlotsCommand,
        )
        if not all(isinstance(command, allowed_commands) for command in task.commands):
            return self._reject(ClarifyReason.INVALID_TASK_COMMANDS)

        start_commands = [
            command for command in task.commands if isinstance(command, StartFlowCommand)
        ]
        if len(start_commands) > 1:
            return self._reject(ClarifyReason.MULTIPLE_TASK_FLOWS)

        start_flow_id = start_commands[0].flow if start_commands else None
        if start_flow_id and self._user_flow(flows_list, start_flow_id) is None:
            return self._reject(ClarifyReason.UNKNOWN_TASK_FLOW)

        resume_commands = [
            command for command in task.commands if isinstance(command, ResumeFlowCommand)
        ]
        for command in resume_commands:
            if command.flow and self._user_flow(flows_list, command.flow) is None:
                return self._reject(ClarifyReason.UNKNOWN_RESUME_FLOW)

        # set_slots 必须绑定到一个明确的业务 Flow：
        # 同轮 start_flow > 当前 active_flow > 精确 resume_flow。
        target_flow_id = start_flow_id or active_flow
        if target_flow_id is None:
            for command in resume_commands:
                if command.flow:
                    target_flow_id = command.flow
                    break

        set_slot_commands = [
            command for command in task.commands if isinstance(command, SetSlotsCommand)
        ]
        if set_slot_commands:
            target_flow = self._user_flow(flows_list, target_flow_id)
            if target_flow is None:
                return self._reject(ClarifyReason.INVALID_TASK_SLOTS)

            allowed_slot_names = set(target_flow.slots.keys())
            for command in set_slot_commands:
                if not command.slots:
                    return self._reject(ClarifyReason.INVALID_TASK_SLOTS)
                if not set(command.slots.keys()).issubset(allowed_slot_names):
                    return self._reject(ClarifyReason.INVALID_TASK_SLOTS)

        return TurnPlanValidatedResult(valid=True)

    def _validate_knowledge_track(
        self,
        knowledge: KnowledgeTurnPlan,
        focused_object: dict[str, Any] | None,
        knowledge_intents: dict[str, KnowledgeIntent],
    ) -> TurnPlanValidatedResult:
        if not knowledge.intents:
            return self._reject(ClarifyReason.MISSING_KNOWLEDGE_INTENT)

        for intent_id in knowledge.intents:
            knowledge_meta = knowledge_intents.get(intent_id)
            if knowledge_meta is None:
                return self._reject(ClarifyReason.UNKNOWN_KNOWLEDGE_INTENT)

            requires_object = knowledge_meta.requires_object
            if requires_object is not None:
                if focused_object is None or focused_object.get("type") != requires_object:
                    return self._reject(ClarifyReason.MISSING_FOCUSED_OBJECT)

        return TurnPlanValidatedResult(valid=True)
