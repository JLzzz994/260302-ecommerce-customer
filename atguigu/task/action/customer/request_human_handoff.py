from atguigu.domain.messages import BotMessage
from atguigu.task.action.base import Action, ActionResult


class ActionRequestHumanHandoff(Action):
    """把文本意图“转人工”真正接到 WebSocket 人工坐席队列。

    TransferManager 仍负责 machine -> queue -> human 的会话状态和消息路由；
    Flow 只负责触发该受控动作，不让 LLM 自己修改转接状态。
    """

    name = "action_request_human_handoff"

    async def run(self, action_kwargs, ctx) -> ActionResult:
        sender_id = ctx.user_message.sender_id if ctx.user_message is not None else None
        if not sender_id:
            return ActionResult(messages=[
                BotMessage(text="当前无法识别会话用户，请通过人工客服入口重新发起转接。")
            ])

        # 延迟导入，避免对话核心模块在装配阶段反向依赖 API 路由。
        from atguigu.api.transfer_manager import transfer_manager

        result = await transfer_manager.request_transfer(sender_id)

        if result.result == "queued":
            text = f"已进入人工客服队列，当前排队位置：{result.queue_position}。排队期间仍可以继续使用智能客服。"
        elif result.result == "already_queued":
            text = f"你已经在人工客服队列中，当前排队位置：{result.queue_position}。"
        elif result.result == "already_human":
            text = "当前已经由人工客服接入，我不会再把后续消息交给机器人处理。"
        else:
            text = "当前没有在线人工客服。你可以稍后再次发起转接；涉及退款执行、订单修改等高风险操作，我不会直接执行。"

        return ActionResult(messages=[BotMessage(text=text)])
