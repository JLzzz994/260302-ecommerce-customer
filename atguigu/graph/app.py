"""对话应用门面：封装 LangGraph 每轮输入、interrupt 恢复与历史切片。"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from atguigu.domain.messages import (
    BotMessage,
    FocusedObject,
    MessageType,
    ProcessResult,
    UserMessage,
)


class DialogueApp:

    def __init__(self, graph, checkpointer=None):
        self._graph = graph
        self._checkpointer = checkpointer

    async def chat(self, user_message: UserMessage) -> ProcessResult:
        """处理一轮用户消息：新输入，或恢复正在 interrupt 的业务子图。"""
        sender_id = user_message.sender_id
        config = {"configurable": {"thread_id": sender_id}}

        state_before = await self._graph.aget_state(config)

        if state_before.next:
            # interrupt 恢复不会经过 ingest，所以必须先把这一轮用户回答写回图状态：
            # 1) user_message 供 Action/Responder 读取当前轮；
            # 2) messages 进入持久化历史，后续 Planner 才看得到“订单号是 A001”等补槽话术；
            # 3) 卡片恢复时同步 focused_object。
            updates: dict[str, Any] = {
                "user_message": user_message,
                "messages": [self._to_human_message(user_message)],
            }
            if user_message.type is MessageType.OBJECT and user_message.object is not None:
                updates["focused_object"] = user_message.object.to_dict()

            await self._graph.aupdate_state(config, updates)
            graph_input = Command(resume=self._build_resume_payload(user_message))
        else:
            graph_input = {"user_message": user_message, "messages": []}

        result = await self._graph.ainvoke(graph_input, config)

        # collect 再次 interrupt 时，问题文案在 interrupt payload 里；
        # 将问题补进 messages，既返回前端也进入 checkpoint 历史。
        snapshot_after = await self._graph.aget_state(config)
        pending_question = None
        if snapshot_after.next and snapshot_after.tasks:
            for task in snapshot_after.tasks:
                if task.interrupts:
                    payload = task.interrupts[0].value
                    if (
                        isinstance(payload, dict)
                        and payload.get("question")
                        and not payload.get("silent")
                    ):
                        pending_question = payload["question"]
                    break

        if pending_question:
            await self._graph.aupdate_state(
                config,
                {"messages": [AIMessage(content=pending_question)]},
            )

        final_snapshot = await self._graph.aget_state(config)
        reply_messages = self._slice_new_ai_messages(
            state_before,
            final_snapshot.values.get("messages", []),
        )

        return ProcessResult(
            message_id=user_message.message_id,
            messages=reply_messages,
        )

    def _to_human_message(self, user_message: UserMessage) -> HumanMessage:
        kwargs: dict[str, Any] = {"_turn_message_id": user_message.message_id}
        if user_message.type is MessageType.OBJECT and user_message.object is not None:
            kwargs["object"] = user_message.object.to_dict()
            content = user_message.text or user_message.object.title or ""
        else:
            content = user_message.text or ""

        return HumanMessage(
            content=content,
            additional_kwargs=kwargs,
        )

    def _build_resume_payload(self, user_message: UserMessage) -> dict[str, Any]:
        if user_message.type is MessageType.OBJECT and user_message.object is not None:
            return {
                "object_id": user_message.object.id,
                "text": user_message.text,
            }
        return {"text": user_message.text}

    def _slice_new_ai_messages(self, snapshot, messages) -> list[BotMessage]:
        before = len(snapshot.values.get("messages", [])) if snapshot is not None else 0
        new_messages = messages[before:]

        replies: list[BotMessage] = []
        for message in new_messages:
            if isinstance(message, AIMessage):
                obj = message.additional_kwargs.get("object")
                replies.append(
                    BotMessage(
                        text=message.content,
                        object=FocusedObject.from_dict(obj) if obj else None,
                    )
                )
        return replies

    async def get_history(self, sender_id: str) -> list[dict[str, Any]]:
        config = {"configurable": {"thread_id": sender_id}}
        snapshot = await self._graph.aget_state(config)
        history = []

        for message in snapshot.values.get("messages", []) if snapshot else []:
            is_user = isinstance(message, HumanMessage)
            obj = message.additional_kwargs.get("object")
            history.append({
                "session_id": sender_id,
                "role": "user" if is_user else "bot",
                "text": message.content,
                "object": obj,
            })

        return history
