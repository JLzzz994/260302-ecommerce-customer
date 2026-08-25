"""
对话应用（对 API 层的门面）：封装"每轮交互"的完整生命周期

用法：
    app = DialogueApp(graph, checkpointer)
    reply = await app.chat(sender_id, text="查下订单A001")

职责：
1. 把用户输入转成图输入（HumanMessage / Command(resume=...)）
2. 调 graph.ainvoke（thread_id=sender_id，checkpointer 自动持久化/恢复中断点）
3. 从 messages 通道里切出"本轮新增的回复"返回
4. interrupt 暴露为返回值的一部分（_pending_question），供前端渲染提问
"""

import uuid
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.types import Command

from atguigu.domain.messages import UserMessage, ProcessResult, BotMessage, MessageType, FocusedObject


class DialogueApp:

    def __init__(self, graph, checkpointer=None):
        self._graph = graph
        self._checkpointer = checkpointer

    async def chat(self,
                   user_message: UserMessage) -> ProcessResult:
        """处理一轮用户消息：新消息 or 恢复中断（等槽位的流程）"""
        sender_id = user_message.sender_id
        config = {"configurable": {"thread_id": sender_id}}

        # 本轮输入：文本走新输入；卡片走新输入（focused_object由ingest节点入状态）
        graph_input = {"user_message": user_message, "messages": []}

        # 如果图正处于 interrupt 中断点：把本轮输入作为恢复值
        state_snapshot = await self._graph.aget_state(config)
        if state_snapshot.next:  # 有待恢复节点 = 图停在中断点
            resume_payload = self._build_resume_payload(user_message)
            graph_input = Command(resume=resume_payload)

        result = await self._graph.ainvoke(graph_input, config)

        # interrupt 中断（收集槽位等用户）：从快照取提问文案，作为本轮机器人消息补发
        # （interrupt 抛出时节点部分更新不落盘，提问必须在这里补进 messages）
        snapshot_after = await self._graph.aget_state(config)
        pending_question = None
        if snapshot_after.next and snapshot_after.tasks:
            for task in snapshot_after.tasks:
                if task.interrupts:
                    payload = task.interrupts[0].value
                    if isinstance(payload, dict) and payload.get("question") and not payload.get("silent"):
                        pending_question = payload["question"]
                    break
        if pending_question:
            await self._graph.aupdate_state(config, {"messages": [AIMessage(content=pending_question)]})

        # 提取本轮新增回复：取 messages 里最后一个未标记的 AIMessage 序列
        messages = result.get("messages", [])
        reply_messages: list[BotMessage] = []
        for message in messages:
            if isinstance(message, AIMessage) and not message.additional_kwargs.get("_this_turn"):
                obj = message.additional_kwargs.get("object")
                reply_messages.append(BotMessage(
                    text=message.content,
                    object=FocusedObject.from_dict(obj) if obj else None
                ))

        # 只返回本轮的：与调用前快照对比截取新增（含中断补发的提问）
        final_snapshot = await self._graph.aget_state(config)
        reply_messages = self._slice_new_ai_messages(state_snapshot, final_snapshot.values.get("messages", []))

        return ProcessResult(message_id=user_message.message_id, messages=reply_messages)

    def _build_resume_payload(self, user_message: UserMessage) -> dict[str, Any]:
        """interrupt 恢复值：文本或卡片ID"""
        if user_message.type is MessageType.OBJECT and user_message.object is not None:
            return {"object_id": user_message.object.id, "text": user_message.text}
        return {"text": user_message.text}

    def _slice_new_ai_messages(self, snapshot, messages) -> list[BotMessage]:
        """对比调用前快照，切出本轮新增的 AI 消息"""
        from langchain_core.messages import AIMessage
        before = len(snapshot.values.get("messages", [])) if snapshot is not None else 0
        new_messages = messages[before:]

        replies: list[BotMessage] = []
        for message in new_messages:
            if isinstance(message, AIMessage):
                obj = message.additional_kwargs.get("object")
                replies.append(BotMessage(text=message.content,
                                          object=FocusedObject.from_dict(obj) if obj else None))
        return replies

    async def get_history(self, sender_id: str) -> list[dict[str, Any]]:
        """图状态里的完整对话历史（messages 通道）"""
        config = {"configurable": {"thread_id": sender_id}}
        snapshot = await self._graph.aget_state(config)
        history = []
        for message in snapshot.values.get("messages", []) if snapshot else []:
            from langchain_core.messages import HumanMessage
            is_user = isinstance(message, HumanMessage)
            obj = message.additional_kwargs.get("object")
            history.append({
                "session_id": sender_id,
                "role": "user" if is_user else "bot",
                "text": message.content,
                "object": obj,
            })
        return history
