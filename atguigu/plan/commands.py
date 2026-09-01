"""TurnPlanner 支持的四类确定性命令。"""

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Command:
    command: str

    @staticmethod
    def from_dict(command_data: dict[str, Any]) -> "Command":
        """把模型 JSON 转成命令对象。

        未知 command 不直接抛 KeyError，而是先保留成基类 Command，
        交给 TurnPlanValidator 统一判定为 INVALID_TASK_COMMANDS。
        """
        command = str(command_data.get("command") or "")
        clz = COMMAND_TO_CLASS.get(command)
        if clz is None:
            return Command(command=command)

        if clz is StartFlowCommand:
            return StartFlowCommand(command=command, flow=str(command_data.get("flow") or ""))
        if clz is ResumeFlowCommand:
            return ResumeFlowCommand(command=command, flow=command_data.get("flow"))
        if clz is CancelFlowCommand:
            return CancelFlowCommand(command=command, flow=command_data.get("flow"))
        if clz is SetSlotsCommand:
            raw_slots = command_data.get("slots")
            return SetSlotsCommand(
                command=command,
                slots=raw_slots if isinstance(raw_slots, dict) else {},
            )
        return Command(command=command)


@dataclass(slots=True)
class StartFlowCommand(Command):
    flow: str


@dataclass(slots=True)
class SetSlotsCommand(Command):
    slots: dict[str, Any]


@dataclass(slots=True)
class CancelFlowCommand(Command):
    flow: str | None = None


@dataclass(slots=True)
class ResumeFlowCommand(Command):
    flow: str | None = None


COMMAND_TO_CLASS: dict[str, type[Command]] = {
    "start_flow": StartFlowCommand,
    "resume_flow": ResumeFlowCommand,
    "cancel_flow": CancelFlowCommand,
    "set_slots": SetSlotsCommand,
}
