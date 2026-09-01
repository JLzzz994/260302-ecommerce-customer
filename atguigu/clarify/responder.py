import json
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.prompt import PromptTemplate

from atguigu.domain.messages import BotMessage
from atguigu.graph.context import TurnContext
from atguigu.infrastructure.llm import llm_client
from atguigu.plan.turn_plan import ClarifyReason
from atguigu.prompt.loader import load_prompt_template


class ClarifyResponder:
    """把确定性 Validator 的原因码转成用户可理解的澄清话术。"""

    async def respond(
        self,
        reason: ClarifyReason,
        ctx: TurnContext,
    ) -> list[BotMessage]:
        prompt_inputs = self._build_responder_prompt_inputs(reason, ctx)
        return await self._invoke(prompt_inputs)

    def _build_responder_prompt_inputs(
        self,
        reason: ClarifyReason,
        ctx: TurnContext,
    ) -> dict[str, Any]:
        return {
            "user_message": ctx.user_message_text(),
            "history": ctx.history_text(last_n=10),
            "focused_object": (
                json.dumps(ctx.focused_object, ensure_ascii=False)
                if ctx.focused_object is not None
                else "null"
            ),
            "clarify_message": self._build_base_response(reason, ctx),
            "reason": reason.value,
        }

    async def _invoke(self, prompt_inputs: dict[str, Any]) -> list[BotMessage]:
        prompt_template_str = load_prompt_template("clarify_respond")
        prompt_template = PromptTemplate.from_template(
            template=prompt_template_str,
            template_format="jinja2",
        )
        chain = prompt_template | llm_client | StrOutputParser()
        result = await chain.ainvoke(prompt_inputs)
        return [BotMessage(text=result)]

    def _build_base_response(
        self,
        reason: ClarifyReason,
        ctx: TurnContext,
    ) -> str:
        if reason is ClarifyReason.MULTIPLE_TRACKS:
            return "你这次同时提到了多个方向。我们先处理一个，你想先办业务还是先咨询信息呢？"

        if reason is ClarifyReason.MULTIPLE_TASK_FLOWS:
            return (
                "你一次提到了好几件要办的事。我们一件一件来，你想先处理哪一件？"
                "剩下的我会在当前流程结束后提醒你。"
            )

        if reason is ClarifyReason.MISSING_FOCUSED_OBJECT:
            return "请先选择对应的订单或商品，我再继续帮你查询。"

        if reason is ClarifyReason.MISSING_KNOWLEDGE_INTENT:
            return "你是想了解商品、订单、物流配送，还是退换货和平台规则呢？"

        if reason is ClarifyReason.UNKNOWN_KNOWLEDGE_INTENT:
            return "这个知识类型当前不在可查询范围内。你可以改问商品、订单、物流、退换货或平台规则。"

        if reason is ClarifyReason.MISSING_TRACK:
            return "我还没判断清楚你的诉求。请说明是要办理订单/物流/售后业务，还是咨询相关信息。"

        if reason is ClarifyReason.MISSING_TASK_COMMANDS:
            return "我识别到了业务诉求，但还缺少可执行步骤。请再说具体一点，比如查订单、查物流、售后建议或转人工。"

        if reason is ClarifyReason.INVALID_TASK_COMMANDS:
            return "当前规划结果不符合系统允许的操作范围，请重新描述你要处理的业务。"

        if reason is ClarifyReason.UNKNOWN_TASK_FLOW:
            return "这个业务流程当前不开放。你可以处理订单查询、物流查询、售后/退款建议、商品推荐或人工转接。"

        if reason is ClarifyReason.UNKNOWN_RESUME_FLOW:
            return "没有找到你要恢复的业务流程。你可以直接说要继续哪一项订单、物流或售后处理。"

        if reason is ClarifyReason.INVALID_TASK_SLOTS:
            return "你提供的信息与当前业务步骤不匹配，我不会把它写入流程状态。请按当前提示补充对应信息。"

        if reason is ClarifyReason.OBJECT_REQUIRES_INTENT:
            focused_object = ctx.focused_object
            if focused_object is not None and focused_object.get("type") == "order":
                return "我已经收到这个订单了。你想查订单状态、查物流，还是生成售后/退款处理建议？"
            if focused_object is not None and focused_object.get("type") == "product":
                return "我已经收到这个商品了。你想查看商品信息，还是推荐相似商品？"

        return "我还需要再确认一下你的意思，你可以换个更具体的说法告诉我。"
