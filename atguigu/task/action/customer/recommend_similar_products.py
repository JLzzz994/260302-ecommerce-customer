from atguigu.domain.messages import BotMessage
from atguigu.task.action.base import Action, ActionResult
from atguigu.task.action.customer.shared import fetch_product, fetch_similar_products


class ActionRecommendSimilarProducts(Action):
    name = "action_recommend_similar_products"

    async def run(self, action_kwargs, ctx) -> ActionResult:
        product_id = ctx.slots.get("product_id")
        label = product_id or "当前商品"

        payload = await fetch_product(product_id)
        if payload:
            label = str(payload.get("title") or "").strip() or label

        candidates = await fetch_similar_products(product_id)
        if not candidates:
            return ActionResult(messages=[BotMessage(
                text=f"暂时没有查到“{label}”的可用相似商品候选。你可以换一个商品，或转人工客服进一步确认。"
            )])

        lines = []
        for index, item in enumerate(candidates[:3], start=1):
            title = str(item.get("title") or item.get("name") or item.get("product_id") or "候选商品")
            price = item.get("price")
            reason = str(item.get("reason") or item.get("similarity_reason") or "").strip()
            detail = f"{index}. {title}"
            if price is not None:
                detail += f"（¥{price}）"
            if reason:
                detail += f"：{reason}"
            lines.append(detail)

        text = f"基于“{label}”为你找到以下相似商品：\n" + "\n".join(lines)
        return ActionResult(messages=[BotMessage(text=text)])
