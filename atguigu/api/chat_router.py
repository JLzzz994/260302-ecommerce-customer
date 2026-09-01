import uuid

from fastapi import APIRouter

from atguigu.api.dependencies import DialogueServiceDep
from atguigu.api.schemas import (
    ChatBotMessage,
    ChatHistoryResponse,
    ChatObject,
    ChatRequest,
    ChatResponse,
)
from atguigu.domain.messages import FocusedObject, MessageType, ProcessResult, UserMessage

router = APIRouter()


@router.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest, service: DialogueServiceDep):
    user_message = _build_user_message(chat_request)
    process_result: ProcessResult = await service.process_message(user_message)
    return _build_chat_response(process_result)


def _build_user_message(chat_request: ChatRequest) -> UserMessage:
    return UserMessage(
        sender_id=chat_request.sender_id,
        message_id=chat_request.message_id or str(uuid.uuid4()),
        type=MessageType.OBJECT if chat_request.object is not None else MessageType.TEXT,
        text=chat_request.text,
        object=(
            FocusedObject(
                id=chat_request.object.id,
                type=chat_request.object.type,
                title=chat_request.object.title,
                attributes=chat_request.object.attributes,
            )
            if chat_request.object is not None
            else None
        ),
    )


def _build_chat_response(process_result: ProcessResult) -> ChatResponse:
    return ChatResponse(
        message_id=process_result.message_id,
        messages=[
            ChatBotMessage(
                text=bot_message.text,
                object=(
                    ChatObject(
                        id=bot_message.object.id,
                        type=bot_message.object.type,
                        title=bot_message.object.title,
                        attributes=bot_message.object.attributes,
                    )
                    if bot_message.object is not None
                    else None
                ),
            )
            for bot_message in process_result.messages
        ],
    )


@router.get("/api/chat/history", response_model=ChatHistoryResponse)
async def chat_history_endpoint(sender_id: str, service: DialogueServiceDep):
    chat_history_messages = await service.get_chat_history(sender_id)
    return ChatHistoryResponse(sender_id=sender_id, messages=chat_history_messages)
