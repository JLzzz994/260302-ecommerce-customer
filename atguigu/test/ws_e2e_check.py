"""
自动化端到端验证：模拟"用户 + 坐席"两个 ws 客户端，跑完整转人工闭环。
前置：先启动演示服务  uv run python -m atguigu.test.ws_standalone_demo
运行：uv run python -m atguigu.test.ws_e2e_check
（也可以当作面试前的"回归自检"，不依赖浏览器）
"""

import asyncio
import json

import websockets

BASE = "ws://127.0.0.1:18083"


async def recv_until(ws, expected_type: str, timeout: float = 5.0) -> dict:
    """循环收消息直到出现指定 type（跳过 heartbeat 等），超时抛异常"""
    async with asyncio.timeout(timeout):
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("type") == expected_type:
                return msg


async def main():
    async with websockets.connect(f"{BASE}/ws/agent/a1") as agent, \
               websockets.connect(f"{BASE}/ws/user/u1") as user:

        ready = await recv_until(agent, "agent_ready")  # 跳过前面的 connected
        print("1. 坐席上线 ✓ agent_ready")

        assert json.loads(await user.recv())["type"] == "connected"
        print("2. 用户连接 ✓ connected")

        # machine 模式：机器人回复
        await user.send(json.dumps({"type": "chat", "text": "你好"}))
        bot = await recv_until(user, "bot_messages")
        assert "机器人" in bot["messages"][0]["text"]
        print(f"3. 机器模式聊天 ✓ 机器人回复: {bot['messages'][0]['text'][:20]}...")

        # 转人工排队
        await user.send(json.dumps({"type": "transfer_request"}))
        pos = await recv_until(user, "queue_position")
        assert pos["position"] == 1
        upd = await recv_until(agent, "queue_update")
        assert upd["queue"][0]["sender_id"] == "u1"
        print("4. 转人工排队 ✓ 用户第1位，坐席收到 queue_update")

        # 排队期间用户消息仍走机器人（不再进引擎的设计选择）
        await user.send(json.dumps({"type": "chat", "text": "还在吗"}))
        await recv_until(user, "bot_messages")
        print("5. 排队期间仍可和机器人聊 ✓")

        # 坐席接入
        await agent.send(json.dumps({"type": "accept", "sender_id": "u1"}))
        acc_user = await recv_until(user, "transfer_accepted")
        acc_agent = await recv_until(agent, "transfer_accepted")
        assert acc_user["agent_id"] == "a1" and acc_agent["sender_id"] == "u1"
        print("6. 坐席接入 ✓ 双方均收到 transfer_accepted")

        # human 模式：用户消息 → 坐席（不进机器人）
        await user.send(json.dumps({"type": "chat", "text": "我要投诉订单888"}))
        got = await recv_until(agent, "user_message")
        assert got["text"] == "我要投诉订单888"
        print("7. 人工模式路由 ✓ 用户消息转发到了坐席")

        # 坐席回复 → 服务端主动推给用户
        await agent.send(json.dumps({"type": "agent_chat", "sender_id": "u1", "text": "您好，请问具体什么问题？"}))
        reply = await recv_until(user, "agent_message")
        assert reply["text"].startswith("您好")
        print("8. 坐席回复推送 ✓ 用户收到 agent_message")

        # 结束会话，回到机器模式
        await agent.send(json.dumps({"type": "close_session", "sender_id": "u1"}))
        closed_u = await recv_until(user, "session_closed")
        await recv_until(agent, "session_closed")
        assert closed_u["reason"] == "agent_closed"
        print("9. 结束会话 ✓ 双方收到 session_closed，用户回到机器模式")

        # 结束后用户消息重新走机器人
        await user.send(json.dumps({"type": "chat", "text": "谢谢"}))
        await recv_until(user, "bot_messages")
        print("10. 回到机器模式 ✓ 机器人恢复服务")

    print("\n全部 10 步通过 ✅ 转人工全链路（连接/排队/接入/路由/推送/关闭/清理）工作正常")


if __name__ == "__main__":
    asyncio.run(main())
