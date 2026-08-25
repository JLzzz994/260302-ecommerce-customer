from atguigu.graph.context import TurnContext
from atguigu.knowledge.providers.base import KnowledgeChunk
from atguigu.prompt.loader import load_prompt_template
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from atguigu.infrastructure.llm import llm_client
from atguigu.domain.messages import BotMessage


class KnowledgeResponder:
    async def respond(self,
                      chunks: list[KnowledgeChunk],
                      ctx: TurnContext) -> list[BotMessage]:
        # 1. 加载提示词模版内容
        prompt_template_str = load_prompt_template("knowledge_respond")

        # 2. 实例化提示词模版对象
        prompt_template = PromptTemplate.from_template(template=prompt_template_str, template_format="jinja2")

        # 3. 定义chain
        chain = prompt_template | llm_client | StrOutputParser()

        # 4. 调用
        result = await chain.ainvoke({
            "user_message": ctx.user_message_text(),
            "history": ctx.history_text(last_n=10),
            "knowledge_content": "\n\n".join([chunk.content for chunk in chunks])
        })

        return [BotMessage(text=result)]
