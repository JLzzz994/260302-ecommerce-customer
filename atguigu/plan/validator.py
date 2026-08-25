from typing import Any

from atguigu.knowledge.intents import KnowledgeIntent
from atguigu.plan.turn_plan import TurnPlan, TurnPlanValidatedResult, ClarifyReason, TaskTurnPlan, KnowledgeTurnPlan
from atguigu.task.flows.flows import FlowsList
from atguigu.plan.commands import StartFlowCommand, ResumeFlowCommand, CancelFlowCommand, SetSlotsCommand


class TurnPlanValidator:

    def validate(self,
                 turn_plan: TurnPlan,
                 focused_object: dict[str, Any] | None,
                 flows_list: FlowsList,
                 knowledge_intents: dict[str, KnowledgeIntent]) -> TurnPlanValidatedResult:
        """
        职责：校验 turn_plan 轮次结果对象
        Args:
            turn_plan: 路由模型产出的轮次规划
            focused_object: 图状态里的卡片（dict 或 None）——知识轨道 requires_object 校验用
            flows_list:
            knowledge_intents:

        Returns: TurnPlanValidatedResult：校验结果
        """

        # 1. 轨道层面校验（外部）
        selected_tracks = turn_plan.activated_tracks()

        # 1.1 是否没有命中任何轨道
        if not selected_tracks:
            return self._reject(ClarifyReason.MISSING_TRACK)

        # 1.2 是否命中了多条轨道
        if len(selected_tracks) > 1:
            return self._reject(ClarifyReason.MULTIPLE_TRACKS)

        # 2. 轨道内部校验（内部）三条轨道[闲聊不需要校验]
        selected_track = selected_tracks[0]
        if selected_track == "task":
            return self._validate_task_track(turn_plan.task, flows_list)
        elif selected_track == "knowledge":
            return self._validate_knowledge_track(turn_plan.knowledge, focused_object, knowledge_intents)
        else:
            return TurnPlanValidatedResult(valid=True)

    def _reject(self, reason: ClarifyReason) -> TurnPlanValidatedResult:
        return TurnPlanValidatedResult(valid=False, reason=reason)

    def _validate_task_track(self,
                             task: TaskTurnPlan,
                             flows_list: FlowsList) -> TurnPlanValidatedResult:
        """
        task轨道内部校验四层：
        1. 校验commands是否是空列表
        2. 校验commands中的命令类型是否是系统现在支持的四种命令类型
        3. 校验commands中是否有多个开启业务流程的命令 [StartFlowCommand]
        4. 校验这一个业务流程是否存在【flow_id---->Flows查询】
        """
        # 1. 校验commands是否是空列表
        if not task.commands:
            return self._reject(reason=ClarifyReason.MISSING_TASK_COMMANDS)

        # 2. 校验命令类型是否是系统支持的四种
        allowed_commands = (StartFlowCommand, ResumeFlowCommand, CancelFlowCommand, SetSlotsCommand)
        if not all(isinstance(command, allowed_commands) for command in task.commands):
            return self._reject(reason=ClarifyReason.INVALID_TASK_COMMANDS)

        # 3. 校验是否多个 start_flow
        start_flow_cmd = [command for command in task.commands if isinstance(command, StartFlowCommand)]
        if len(start_flow_cmd) > 1:
            return self._reject(reason=ClarifyReason.MULTIPLE_TASK_FLOWS)

        # 4. 校验业务流程存在性（没有 StartFlowCommand 不拒绝：可能是恢复/取消/设槽位）
        if start_flow_cmd:
            flow = flows_list.get_flow_by_flow_id(start_flow_cmd[0].flow)
            if flow is None:
                return self._reject(reason=ClarifyReason.UNKNOWN_TASK_FLOW)

        return TurnPlanValidatedResult(valid=True)

    def _validate_knowledge_track(self,
                                  knowledge: KnowledgeTurnPlan,
                                  focused_object: dict[str, Any] | None,
                                  knowledge_intents: dict[str, KnowledgeIntent]) -> TurnPlanValidatedResult:
        """
        职责：校验知识轨道内部的知识意图是否要求卡片对象、卡片类型是否匹配
        """
        for intent_id in knowledge.intents:
            knowledge_meta = knowledge_intents[intent_id]
            requires_object = knowledge_meta.requires_object

            if requires_object is not None:
                if focused_object is None or focused_object.get("type") != requires_object:
                    return self._reject(reason=ClarifyReason.MISSING_FOCUSED_OBJECT)

        return TurnPlanValidatedResult(valid=True)
