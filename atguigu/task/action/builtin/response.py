from typing import Any
from jinja2 import Template

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from atguigu.infrastructure.llm import llm_client
from atguigu.domain.messages import BotMessage
from atguigu.graph.context import TurnContext
from atguigu.task.action.base import Action, ActionResult


class ActionResponse(Action):
    name = "action_response"

    async def run(self,
                  action_kwargs: dict[str, Any],
                  ctx: TurnContext
                  ) -> ActionResult:
        """
        职责：负责将YAML中的响应内容，获取到返回
        响应内容：有占位。双花括号：交给jinja2模版引擎（渲染）
        """

        # 1. 获取响应的模式
        action_response_mode = action_kwargs.get('mode', 'static')

        # 2. 判断模式
        if action_response_mode == "rephrase":
            # a) 渲染要响应的内容
            rendered_text = self._render(action_kwargs['text'], ctx)
            # b) 调用llm润色
            rewritten = await self._call_llm(ctx, action_kwargs['prompt'], rendered_text)
            return ActionResult(messages=[BotMessage(text=rewritten)])
        elif action_response_mode == "generate":
            # 只有提示词，从0生成
            rewritten = await self._call_llm(ctx, action_kwargs['prompt'])
            return ActionResult(messages=[BotMessage(text=rewritten)])
        else:
            # "static"：渲染后直接返回
            rendered_text = self._render(action_kwargs['text'], ctx)
            return ActionResult(messages=[BotMessage(text=rendered_text)])

    def _render(self, response_text: str, ctx: TurnContext) -> str:
        template = Template(response_text)
        return template.render(slots=ctx.slots, context=ctx.flow_context or {})

    async def _call_llm(self,
                        ctx: TurnContext,
                        prompt: str,
                        rendered_text: str = "") -> str:
        prompt_template = PromptTemplate.from_template(template=prompt)
        chain = prompt_template | llm_client | StrOutputParser()
        return await chain.ainvoke({
            "history": ctx.history_text(last_n=5),
            "user_message": ctx.user_message_text(),
            "current_response": rendered_text
        })
