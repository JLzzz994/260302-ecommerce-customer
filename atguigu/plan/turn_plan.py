from dataclasses import dataclass
from enum import Enum
from typing import Any

from atguigu.plan.commands import Command


@dataclass(slots=True)
class TaskTurnPlan:
    commands: list[Command]

    @classmethod
    def from_dict(cls, task_data: dict[str, Any]) -> "TaskTurnPlan":
        raw_commands = task_data.get("commands")
        if not isinstance(raw_commands, list):
            raw_commands = []
        return cls(
            commands=[
                Command.from_dict(item if isinstance(item, dict) else {})
                for item in raw_commands
            ]
        )


@dataclass(slots=True)
class KnowledgeTurnPlan:
    intents: list[str]

    @classmethod
    def from_dict(cls, knowledge_data: dict[str, Any]) -> "KnowledgeTurnPlan":
        raw = knowledge_data.get("intents")
        intents = [str(item) for item in raw] if isinstance(raw, list) else []
        return cls(intents=intents)


@dataclass(slots=True)
class ChitChatTurnPlan:
    chat: str

    @classmethod
    def from_dict(cls, chat_data: dict[str, Any]) -> "ChitChatTurnPlan":
        return cls(chat=str(chat_data.get("chat") or ""))


@dataclass(slots=True)
class TurnPlan:
    task: TaskTurnPlan | None = None
    knowledge: KnowledgeTurnPlan | None = None
    chitchat: ChitChatTurnPlan | None = None

    @classmethod
    def from_dict(cls, turn_plan_data: dict[str, Any]) -> "TurnPlan":
        if not isinstance(turn_plan_data, dict):
            turn_plan_data = {}
        task_data = turn_plan_data.get("task")
        knowledge_data = turn_plan_data.get("knowledge")
        chitchat_data = turn_plan_data.get("chitchat")
        return cls(
            task=TaskTurnPlan.from_dict(task_data) if isinstance(task_data, dict) else None,
            knowledge=KnowledgeTurnPlan.from_dict(knowledge_data) if isinstance(knowledge_data, dict) else None,
            chitchat=ChitChatTurnPlan.from_dict(chitchat_data) if isinstance(chitchat_data, dict) else None,
        )

    def activated_tracks(self) -> list[str]:
        tracks = []
        if self.task is not None:
            tracks.append("task")
        if self.knowledge is not None:
            tracks.append("knowledge")
        if self.chitchat is not None:
            tracks.append("chitchat")
        return tracks


class ClarifyReason(Enum):
    MISSING_TRACK = "missing_track"
    MULTIPLE_TRACKS = "multiple_tracks"

    MISSING_TASK_COMMANDS = "missing_task_commands"
    INVALID_TASK_COMMANDS = "invalid_task_commands"
    MULTIPLE_TASK_FLOWS = "multiple_task_flows"
    UNKNOWN_TASK_FLOW = "unknown_task_flow"
    UNKNOWN_RESUME_FLOW = "unknown_resume_flow"
    INVALID_TASK_SLOTS = "invalid_task_slots"

    MISSING_KNOWLEDGE_INTENT = "missing_knowledge_intent"
    UNKNOWN_KNOWLEDGE_INTENT = "unknown_knowledge_intent"
    MISSING_FOCUSED_OBJECT = "missing_focused_object"

    OBJECT_REQUIRES_INTENT = "object_requires_intent"


@dataclass(slots=True)
class TurnPlanValidatedResult:
    valid: bool
    reason: ClarifyReason | None = None
