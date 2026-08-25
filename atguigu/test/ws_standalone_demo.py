"""
WebSocket 转人工 · 独立演示服务（零外部依赖：不需要数据库、不需要 LLM、不需要 .env）

用途：面试前/面试中随时演示"机器人对话 → 转人工排队 → 坐席接入 → 双向聊天 → 结束回机器人"的完整闭环。
它复用了正式代码里的 ConnectionManager / TransferManager（纯内存组件，无外部依赖），
只是把"机器人回复"换成了本地假机器人，所以一条命令就能跑。

运行：
    uv run python -m atguigu.test.ws_standalone_demo
    （或 python -m atguigu.test.ws_standalone_demo）

然后开两个浏览器标签：
    用户端:  http://127.0.0.1:18083/?role=user&id=u1
    坐席端:  http://127.0.0.1:18083/?role=agent&id=a1

演示脚本（配合页面上右侧的协议日志讲解）：
    1. 用户端连接 → 发几条消息 → 假机器人回复（machine 模式走"引擎"）
    2. 点【转人工】→ 服务端入队 → 坐席端看到"接入 xxx"按钮
    3. 坐席点【接入】→ 用户端收到 transfer_accepted（服务端主动推送！HTTP 做不到的点）
    4. 双向聊天：用户消息被路由给坐席（不再进机器人），坐席回复推给用户
    5. 坐席点【结束该会话】→ 双方收到 session_closed，用户回到机器模式
    6. 加分演示：关掉坐席标签页 → 用户端立刻收到 session_closed（断线清理）
"""

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from atguigu.api.connection_manager import connection_manager
from atguigu.api.transfer_manager import transfer_manager

APP_HOST = "127.0.0.1"
APP_PORT = 18083
DEMO_PAGE = Path(__file__).resolve().parent / "ws_demo_page.html"

app = FastAPI(title="WebSocket 转人工演示")


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回测试页（用户/坐席双角色二合一）"""
    return DEMO_PAGE.read_text(encoding="utf-8")


# ---------------------------- 假机器人（替代 DialogueService） ----------------------------

async def fake_bot_reply(text: str) -> list[str]:
    """演示用的本地'对话引擎'：关键词匹配。真实实现见 ws_router 里的 DialogueService。"""
    await asyncio.sleep(0.3)  # 模拟 LLM 延迟，演示效果更真实
    if any(kw in text for kw in ("退", "退款", "投诉", "人工")):
        return ["这个问题我可以帮您转接人工客服，请点击下方【转人工】按钮～"]
    if "订单" in text:
        return ["已为您查询到订单逻辑（演示环境仅回显）。如需人工核实，请点【转人工】。"]
    return [f"（机器人）收到：{text}。我是演示环境的假机器人，正式环境会调用 LLM 对话引擎。"]


# ---------------------------- 用户端 ws 端点 ----------------------------

@app.websocket("/ws/user/{sender_id}")
async def user_ws(websocket: WebSocket, sender_id: str):
    await websocket.accept()
    await connection_manager.connect_user(sender_id, websocket)

    # 恢复模式（与正式 ws_router 一致：刷新页面后找回排队/人工状态）
    session = transfer_manager.get_or_create_session(sender_id)
    if session.mode == "queue":
        queue = transfer_manager.queue_snapshot()
        position = next((q["position"] for q in queue if q["sender_id"] == sender_id), None)
        await websocket.send_json({"type": "queue_position", "position": position})
    elif session.mode == "human":
        await websocket.send_json({"type": "transfer_accepted", "agent_id": session.agent_id})

    try:
        while True:
            payload: dict[str, Any] = await websocket.receive_json()
            msg_type = payload.get("type")

            if msg_type == "heartbeat":
                await websocket.send_json({"type": "heartbeat_ack"})

            elif msg_type == "transfer_request":
                result = await transfer_manager.request_transfer(sender_id)
                if result.result == "no_agent_online":
                    await websocket.send_json({"type": "error", "detail": "当前没有客服在线，请稍后再试"})
                elif result.result == "already_human":
                    await websocket.send_json({"type": "transfer_accepted", "agent_id": session.agent_id})

            elif msg_type == "cancel_transfer":
                await transfer_manager.cancel_transfer(sender_id)

            elif msg_type == "chat":
                # 核心路由：人工模式转发坐席；机器模式走"引擎"
                delivered = await transfer_manager.route_user_message(sender_id, payload)
                if not delivered:
                    texts = await fake_bot_reply(payload.get("text") or "")
                    await websocket.send_json({"type": "bot_messages",
                                               "messages": [{"text": t, "object": None} for t in texts]})

            else:
                await websocket.send_json({"type": "error", "detail": f"未知的消息类型: {msg_type}"})

    except WebSocketDisconnect:
        await transfer_manager.handle_user_disconnect(sender_id)
    finally:
        connection_manager.disconnect_user(sender_id, websocket)


# ---------------------------- 坐席端 ws 端点 ----------------------------

@app.websocket("/ws/agent/{agent_id}")
async def agent_ws(websocket: WebSocket, agent_id: str):
    await websocket.accept()
    await connection_manager.connect_agent(agent_id, websocket)
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
                    await websocket.send_json({"type": "transfer_accepted", "sender_id": target, "agent_id": agent_id})
                else:
                    await websocket.send_json({"type": "error", "detail": f"接入失败：用户 {target} 不在排队中"})

            elif msg_type == "agent_chat":
                await transfer_manager.agent_chat(agent_id, payload.get("sender_id"), payload.get("text", ""))

            elif msg_type == "close_session":
                await transfer_manager.close_session(payload.get("sender_id"), reason="agent_closed")

            else:
                await websocket.send_json({"type": "error", "detail": f"未知的消息类型: {msg_type}"})

    except WebSocketDisconnect:
        await transfer_manager.handle_agent_disconnect(agent_id)
    finally:
        connection_manager.disconnect_agent(agent_id, websocket)


if __name__ == "__main__":
    import uvicorn

    print(f"""
==========================================================
  WebSocket 转人工演示服务启动中: http://{APP_HOST}:{APP_PORT}

  用户端标签:  http://{APP_HOST}:{APP_PORT}/?role=user&id=u1
  坐席端标签:  http://{APP_HOST}:{APP_PORT}/?role=agent&id=a1

  演示顺序: 用户聊天 -> 转人工 -> 坐席接入 -> 双向聊天 -> 结束
==========================================================
""")
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
