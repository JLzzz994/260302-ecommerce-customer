"""
WebSocket 路由：转人工（人机协同）

两个端点：
- /ws/user/{sender_id}   用户端（浏览器聊天窗口）
- /ws/agent/{agent_id}   客服坐席端（坐席工作台）

与现有 HTTP 接口 POST /api/chat 的关系：
- 机器人模式下的处理逻辑完全复用 DialogueService（读状态→引擎→存状态）；
  区别只是"传输层"从 HTTP 一问一答，换成 ws 双向长连接。
- 人工模式下用户消息不再进引擎，而是原样转发给坐席；坐席回复由服务端主动推给用户。
  ——这就是 HTTP 做不到、必须用 WebSocket 的原因（服务端要能主动推送）。

消息协议（JSON，双方都要带 type 字段，详见 docs/websocket-转人工面试手册.md）：
用户→服务端: chat / transfer_request / cancel_transfer / heartbeat
坐席→服务端: accept / agent_chat / close_session / heartbeat
服务端→双方: connected / bot_messages / queue_position / transfer_accepted /
             user_message / agent_message / session_closed / queue_update / heartbeat_ack / error
"""

import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from atguigu.api.chat_router import _build_chat_response, _build_user_message
from atguigu.api.connection_manager import connection_manager
from atguigu.api.dependencies import get_dialogue_app
from atguigu.api.schemas import ChatRequest
from atguigu.api.transfer_manager import transfer_manager
from atguigu.domain.messages import ProcessResult
from atguigu.services.dialogue_service import DialogueService

router = APIRouter()

# 图应用由 app.lifespan 装配的全局单例（编译一次，无并发可变状态）；
# checkpointer 连接池同样全局共享，ws 长连接不再需要"按需建 DB session"的旧模式
_dialogue_service = None


def _get_dialogue_service() -> DialogueService:
    global _dialogue_service
    if _dialogue_service is None:
        _dialogue_service = DialogueService(get_dialogue_app())
    return _dialogue_service


async def _send_error(websocket: WebSocket, detail: str):
    await websocket.send_json({"type": "error", "detail": detail})


# ============================== 用户端 ==============================

@router.websocket("/ws/user/{sender_id}")
async def user_ws(websocket: WebSocket, sender_id: str):
    await websocket.accept()                        # 1. 完成 ws 握手（101 Switching Protocols）
    await connection_manager.connect_user(sender_id, websocket)  # 2. 登记到在线连接表

    # 3. 恢复会话模式：用户刷新页面重连后，如果之前在排队/人工中，把状态续上
    session = transfer_manager.get_or_create_session(sender_id)
    if session.mode == "queue":
        queue = transfer_manager.queue_snapshot()
        position = next((item["position"] for item in queue if item["sender_id"] == sender_id), None)
        await websocket.send_json({"type": "queue_position", "position": position})
    elif session.mode == "human":
        await websocket.send_json({"type": "transfer_accepted", "agent_id": session.agent_id})

    try:
        # 4. 消息循环：阻塞在这里等客户端的下一条消息（这就是"长连接"）
        while True:
            payload: dict[str, Any] = await websocket.receive_json()
            msg_type = payload.get("type")

            if msg_type == "heartbeat":
                # 应用层心跳：探测"业务上"的存活；协议层心跳由 ping/pong 帧承担（面试区分点）
                await websocket.send_json({"type": "heartbeat_ack"})

            elif msg_type == "transfer_request":
                result = await transfer_manager.request_transfer(sender_id)
                if result.result == "no_agent_online":
                    await _send_error(websocket, "当前没有客服在线，请稍后再试")
                elif result.result == "already_human":
                    await websocket.send_json({"type": "transfer_accepted", "agent_id": session.agent_id})

            elif msg_type == "cancel_transfer":
                await transfer_manager.cancel_transfer(sender_id)

            elif msg_type == "chat":
                # 5. 消息路由：人工模式 → 转发给坐席；否则走对话引擎（复用 HTTP 版逻辑）
                delivered = await transfer_manager.route_user_message(sender_id, payload)
                if not delivered:
                    chat_request = ChatRequest(
                        sender_id=sender_id,
                        message_id=str(uuid.uuid4()),
                        text=payload.get("text"),
                        object=payload.get("object"),
                    )
                    user_message = _build_user_message(chat_request)
                    process_result: ProcessResult = await _get_dialogue_service().process_message(user_message)
                    chat_response = _build_chat_response(process_result)
                    await websocket.send_json({
                        "type": "bot_messages",
                        "messages": [m.model_dump() for m in chat_response.messages]
                    })

            else:
                await _send_error(websocket, f"未知的消息类型: {msg_type}")

    except WebSocketDisconnect:
        # 6. 客户端断开（关页面/断网）：清理转人工状态，避免"幽灵会话"
        await transfer_manager.handle_user_disconnect(sender_id)
    finally:
        connection_manager.disconnect_user(sender_id, websocket)


# ============================== 坐席端 ==============================

@router.websocket("/ws/agent/{agent_id}")
async def agent_ws(websocket: WebSocket, agent_id: str):
    await websocket.accept()
    await connection_manager.connect_agent(agent_id, websocket)

    # 坐席上线即推送当前排队列表（工作台首屏数据）
    await websocket.send_json({"type": "agent_ready", "queue": transfer_manager.queue_snapshot()})

    try:
        while True:
            payload: dict[str, Any] = await websocket.receive_json()
            msg_type = payload.get("type")

            if msg_type == "heartbeat":
                await websocket.send_json({"type": "heartbeat_ack"})

            elif msg_type == "accept":
                target = payload.get("sender_id")
                ok = await transfer_manager.agent_accept(agent_id, target)
                if ok:
                    # 也要通知坐席自己接入成功（工作台才知道当前服务谁）
                    await websocket.send_json({"type": "transfer_accepted", "sender_id": target, "agent_id": agent_id})
                else:
                    await _send_error(websocket, f"接入失败：用户 {target} 不在排队中（可能已被其他坐席接入）")

            elif msg_type == "agent_chat":
                target = payload.get("sender_id")
                ok = await transfer_manager.agent_chat(
                    agent_id,
                    target,
                    payload.get("text", ""),
                )
                if not ok:
                    await _send_error(
                        websocket,
                        f"发送失败：用户 {target} 当前未绑定给坐席 {agent_id}",
                    )

            elif msg_type == "close_session":
                target = payload.get("sender_id")
                ok = await transfer_manager.agent_close_session(agent_id, target)
                if not ok:
                    await _send_error(
                        websocket,
                        f"结束失败：用户 {target} 当前未绑定给坐席 {agent_id}",
                    )

            else:
                await _send_error(websocket, f"未知的消息类型: {msg_type}")

    except WebSocketDisconnect:
        await transfer_manager.handle_agent_disconnect(agent_id)
    finally:
        connection_manager.disconnect_agent(agent_id, websocket)
