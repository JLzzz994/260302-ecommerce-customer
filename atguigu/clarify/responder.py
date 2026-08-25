import json
from typing import Any

from atguigu.domain.messages import BotMessage
from atguigu.graph.context import TurnContext
from atguigu.plan.turn_plan import ClarifyReason
from atguigu.prompt.loader import load_prompt_template
from atguigu.infrastructure.llm import llm_client

from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class ClarifyResponder:
    async def respond(self,
                      reason: ClarifyReason,
                      ctx: TurnContext) -> list[BotMessage]:
        """
        根据校验结果对象的原因码，利用LLM 来润色澄清回复
        """
        # 1. 构建澄清话术需要的提示词模版变量值
        prompt_inputs = self._build_responder_prompt_inputs(reason, ctx)

        # 2. 格式化模版，调用LLM
        bot_messages = await self._invoke(prompt_inputs)

        # 3. 返回
        return bot_messages

    def _build_responder_prompt_inputs(self,
                                       reason: ClarifyReason,
                                       ctx: TurnContext) -> dict[str, Any]:
        reason_str = reason.value
        clarify_message_str = self._build_base_response(reason, ctx)

        return {
            "user_message": ctx.user_message_text(),
            "history": ctx.history_text(last_n=10),
            "focused_object": json.dumps(ctx.focused_object, ensure_ascii=False) if ctx.focused_object is not None else "null",
            "clarify_message": clarify_message_str,
            "reason": reason_str,
        }

    async def _invoke(self, prompt_inputs: dict[str, Any]) -> list[BotMessage]:
        # 1. 加载提示词模版
        prompt_template_str = load_prompt_template("clarify_respond")

        # 2. 实例化提示词模版对象
        prompt_template = PromptTemplate.from_template(template=prompt_template_str, template_format="jinja2")

        # 3. 构建chain
        chain = prompt_template | llm_client | StrOutputParser()

        # 4. 执行链
        result = await chain.ainvoke(prompt_inputs)

        # 5. 返回结果
        return [BotMessage(text=result)]

    def _build_base_response(self,
                             reason: ClarifyReason,
                             ctx: TurnContext) -> str:

        if reason is ClarifyReason.MULTIPLE_TRACKS:
            return "你这次同时提到了多个方向。我们先处理一个，你想先办业务还是先咨询信息呢？"

        if reason is ClarifyReason.MULTIPLE_TASK_FLOWS:
            return "你一次提到了好几件要办的事。我们一件一件来，你想先办理哪一件呢？剩下的我记下来了，办完这件会提醒你继续。"

        if reason is ClarifyReason.MISSING_FOCUSED_OBJECT:
            return "请先发送你想咨询的对象，我再继续帮你看。"

        if reason is ClarifyReason.MISSING_KNOWLEDGE_INTENT:
            return "你是想了解商品信息、订单信息，还是售后配送规则呢？"

        if reason is ClarifyReason.MISSING_TRACK:
            return "你是想先处理业务问题，还是先咨询信息呢？"

        if reason is ClarifyReason.MISSING_TASK_COMMANDS:
            return "你这次是想办理什么业务呢？比如查订单、查物流，或者申请退款。"

        if reason is ClarifyReason.OBJECT_REQUIRES_INTENT:
            focused_object = ctx.focused_object
            if focused_object is not None and focused_object.get("type") == "order":
                return "我已经收到这个订单了。你想查订单状态、查物流，还是申请退款呢？"
            if focused_object is not None and focused_object.get("type") == "product":
                return "我已经收到这个商品了。你想了解它的商品信息、发货情况，还是售后相关问题呢？"

        return "我还需要再确认一下你的意思，你可以换个更具体的说法告诉我。"
