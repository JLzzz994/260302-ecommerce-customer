from atguigu.chitchat.responder import ChitChatResponder
from atguigu.domain.messages import BotMessage
from atguigu.graph.context import TurnContext


class ChitChatHandler:

    def __init__(self, chat_responder: ChitChatResponder):
        self._chat_responder = chat_responder

    async def handle(self,
                     chitchat: str,
                     ctx: TurnContext) -> list[BotMessage]:
        return await self._chat_responder.respond_chat(chitchat, ctx)
