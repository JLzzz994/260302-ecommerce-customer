"""
对话服务：API 层与图之间的薄门面（LangGraph 版）

与旧版的区别：状态管理完全交给 LangGraph checkpointer
（thread_id=sender_id 自动按用户隔离/持久化/恢复中断点），
service 不再读写 DialogueState，也不再依赖数据库 repository。
"""

from atguigu.domain.messages import UserMessage, ProcessResult


class DialogueService:

    def __init__(self, dialogue_app):
        self._app = dialogue_app

    async def process_message(self, user_message: UserMessage) -> ProcessResult:
        """一轮对话 = 图的一次 invoke（新输入 或 interrupt 恢复）"""
        return await self._app.chat(user_message)

    async def get_chat_history(self, sender_id: str) -> list[dict]:
        """messages 通道里的完整历史（checkpointer 里的图状态）"""
        return await self._app.get_history(sender_id)
