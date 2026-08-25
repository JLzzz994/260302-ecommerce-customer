from abc import ABC, abstractmethod
from dataclasses import dataclass

from atguigu.graph.context import TurnContext


@dataclass(slots=True)
class KnowledgeChunk:
    content: str


class Provider(ABC):
    """
    抽象基类：知识提供者只依赖 TurnContext（卡片/槽位/当前消息）
    """

    provider_id: str

    @abstractmethod
    async def retrival(self,
                       ctx: TurnContext) -> list[KnowledgeChunk]:
        pass
