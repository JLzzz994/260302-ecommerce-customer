from atguigu.graph.context import TurnContext
from atguigu.domain.messages import BotMessage
from atguigu.knowledge.intents import KnowledgeIntent
from atguigu.knowledge.providers.register import ProviderRegister
from atguigu.knowledge.responder import KnowledgeResponder


class KnowledgeHandler:

    def __init__(self,
                 knowledge_intents: dict[str, KnowledgeIntent],
                 knowledge_responder: KnowledgeResponder,
                 providers_register: ProviderRegister
                 ):
        self.knowledge_intents = knowledge_intents
        self._knowledge_responder = knowledge_responder
        self._providers_register = providers_register

    async def handle(self,
                     ctx: TurnContext,
                     intents: list[str]) -> list[BotMessage]:
        chunks = []
        # 1. 根据LLM提供的知识意图的id(intent),找提供者ID(provider_id)
        provider_ids = self._get_provider_ids_by_intent(intents)

        # 2. 根据提供者ID，查询提供者对象(Provider)，逐个检索
        for provider_id in provider_ids:
            provider = self._providers_register.get_provider(provider_id)
            chunk = await provider.retrival(ctx)
            chunks.extend(chunk)

        # 3. 将所有提供者的检索结果交给 responder 生成回答
        messages = await self._knowledge_responder.respond(chunks, ctx)

        return messages

    def _get_provider_ids_by_intent(self, intents: list[str]) -> list[str]:
        """
        根据意图ID 查询提供者ID
        """
        provider_ids = []
        for intent_id in intents:
            knowledge_intent = self.knowledge_intents[intent_id]
            provider_ids.extend(knowledge_intent.provider_ids)
        return list(set(provider_ids))
