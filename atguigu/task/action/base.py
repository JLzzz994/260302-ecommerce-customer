from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from atguigu.domain.messages import BotMessage
from atguigu.graph.context import TurnContext


@dataclass(slots=True)
class ActionResult:
    messages: list[BotMessage] = field(default_factory=list)  # action_response一定会给messages 但是action_xxx不一定
    slots: dict[str, Any] = field(default_factory=dict)  # action_xxx的run方法一定会给slots


class Action(ABC):
    """
    抽象基类：所有 Action 只依赖 TurnContext（当前消息/历史/槽位/卡片/过场变量）
    """

    name: str

    @abstractmethod
    async def run(self,
                  action_kwargs: dict[str, Any],
                  ctx: TurnContext,
                  ) -> ActionResult:
        pass
