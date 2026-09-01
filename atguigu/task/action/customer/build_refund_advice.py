from atguigu.task.action.base import Action, ActionResult
from atguigu.task.action.customer.shared import fetch_order


class ActionBuildRefundAdvice(Action):
    """只生成售后建议，不执行退款、取消订单或审批等写操作。"""

    name = "action_build_refund_advice"

    async def run(self, action_kwargs, ctx) -> ActionResult:
        order_number = ctx.slots.get("order_number")
        refund_reason = str(ctx.slots.get("refund_reason") or "").strip()
        payload = await fetch_order(order_number)

        if payload is None:
            return ActionResult(slots={
                "refund_advice": "暂时无法核验订单信息，建议转人工客服确认订单、支付、履约和平台售后规则后再处理。",
                "risk_level": "高",
            })

        status = str(payload.get("status_desc") or payload.get("status") or "未知").strip()
        normalized = status.lower()

        if any(keyword in normalized for keyword in ("已发货", "shipped", "已签收", "delivered", "已完成", "completed")):
            risk_level = "高"
            advice = (
                f"订单当前状态为“{status}”。建议先核验物流拦截/退回状态、商品是否已签收、"
                "平台售后时效及退款责任，再由人工客服决定是否发起退款或退货流程。"
            )
        elif any(keyword in normalized for keyword in ("待发货", "已付款", "paid", "processing", "待配货")):
            risk_level = "中"
            advice = (
                f"订单当前状态为“{status}”。建议先确认仓库是否已进入拣货/出库环节，"
                "并结合平台规则与用户原因评估是否可取消或退款；实际操作由人工客服复核。"
            )
        elif any(keyword in normalized for keyword in ("待付款", "unpaid", "未付款")):
            risk_level = "低"
            advice = (
                f"订单当前状态为“{status}”。建议优先核验是否存在实际支付与占用库存，"
                "再按平台规则处理取消或关闭；无需由模型直接执行订单变更。"
            )
        else:
            risk_level = "中"
            advice = (
                f"订单当前状态为“{status}”。建议人工核验支付、履约、退款记录与平台售后规则，"
                "确认风险后再执行实际退款或订单修改。"
            )

        if refund_reason:
            advice += f" 用户反馈原因：{refund_reason}。"

        return ActionResult(slots={
            "refund_advice": advice,
            "risk_level": risk_level,
        })
