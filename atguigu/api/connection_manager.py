"""
WebSocket 连接管理器（ConnectionManager）

核心问题：HTTP 接口是无状态的（一来一回就结束），而 WebSocket 是长连接，
服务端必须"记住"每条连接属于谁（哪个用户/哪个客服坐席），
之后客服发消息时才能找到"对面的用户连接"把消息推过去。

所以需要一个全局注册表：{用户ID: WebSocket} / {坐席ID: WebSocket}

面试话术：
1. 为什么不用 FastAPI 的 Depends 注入到 websocket 路由？
   Depends 里的 get_session 是 `async with` 管理的，请求结束（HTTP）就释放；
   而 ws 长连接可能挂几个小时，把 DB session 绑在长连接上会长期占用连接池。
   所以连接管理只管"纯内存的连接表"，DB 会话在处理每条消息时按需创建、用完即还。
2. 多实例部署（uvicorn 起多个进程/多台机器）时这个内存注册表就不够了——
   用户连在实例 A、客服连在实例 B，需要引入 Redis pub/sub 做跨实例转发（见面试手册）。
"""

import asyncio
from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    """
    统一登记/移除在线的 ws 连接，并提供"按目标发送 JSON 消息"的能力。
    用户（sender_id）和客服坐席（agent_id）分两张表维护。
    """

    def __init__(self):
        # 两张"在线连接注册表"：ID -> WebSocket
        self._user_connections: dict[str, WebSocket] = {}
        self._agent_connections: dict[str, WebSocket] = {}
        # 教学演示用锁；CPython 中单条 dict 原子操作其实不加锁也安全，
        # 但 connect 里有"先查旧连接再覆盖"这种多步逻辑，加锁更严谨
        self._lock = asyncio.Lock()

    # ============================== 用户连接 ==============================

    async def connect_user(self, sender_id: str, websocket: WebSocket):
        """登记用户连接。同一 sender_id 重复连接时采用"顶号"策略：新连接顶掉旧连接。"""
        async with self._lock:
            old = self._user_connections.get(sender_id)
            self._user_connections[sender_id] = websocket

        if old is not None:
            # 旧连接大概率是残留的"半死连接"（比如用户刷新页面没来得及发 FIN）
            try:
                await old.close(code=4001)  # 4001: 自定义业务码，被新连接顶替
            except Exception:
                pass  # 旧连接早已断开，忽略即可

        await websocket.send_json({"type": "connected", "role": "user", "sender_id": sender_id})

    def disconnect_user(self, sender_id: str, websocket: WebSocket):
        """移除用户连接。只有"注册表里登记的还是这条连接"才移除，防止把顶上来的新连接误删。"""
        if self._user_connections.get(sender_id) is websocket:
            self._user_connections.pop(sender_id, None)

    def is_user_online(self, sender_id: str) -> bool:
        return sender_id in self._user_connections

    async def send_to_user(self, sender_id: str, payload: dict[str, Any]) -> bool:
        """
        向指定用户推送一条 JSON 消息（服务端主动推送的核心能力）。
        :return: True=投递成功 False=用户不在线/连接已失效
        发送失败时顺手清理注册表，避免死连接占位。
        """
        websocket = self._user_connections.get(sender_id)
        if websocket is None:
            return False
        try:
            await websocket.send_json(payload)
            return True
        except Exception:
            self.disconnect_user(sender_id, websocket)
            return False

    # ============================== 坐席连接 ==============================

    async def connect_agent(self, agent_id: str, websocket: WebSocket):
        async with self._lock:
            old = self._agent_connections.get(agent_id)
            self._agent_connections[agent_id] = websocket

        if old is not None:
            try:
                await old.close(code=4001)
            except Exception:
                pass

        await websocket.send_json({"type": "connected", "role": "agent", "agent_id": agent_id})

    def disconnect_agent(self, agent_id: str, websocket: WebSocket):
        if self._agent_connections.get(agent_id) is websocket:
            self._agent_connections.pop(agent_id, None)

    async def send_to_agent(self, agent_id: str, payload: dict[str, Any]) -> bool:
        websocket = self._agent_connections.get(agent_id)
        if websocket is None:
            return False
        try:
            await websocket.send_json(payload)
            return True
        except Exception:
            self.disconnect_agent(agent_id, websocket)
            return False

    # ============================== 广播与统计 ==============================

    async def broadcast_to_agents(self, payload: dict[str, Any]):
        """向所有在线坐席广播（典型场景：新用户进入排队，通知所有坐席工作台刷新排队列表）。"""
        for agent_id in list(self._agent_connections.keys()):
            await self.send_to_agent(agent_id, payload)

    def has_online_agents(self) -> bool:
        return len(self._agent_connections) > 0

    def online_agent_ids(self) -> list[str]:
        return list(self._agent_connections.keys())


# 模块级单例：全局只有一张连接注册表，各路由共享。
# FastAPI 的 websocket 路由不方便用 Depends 注入"带状态的单例"，模块级单例最直观。
connection_manager = ConnectionManager()
