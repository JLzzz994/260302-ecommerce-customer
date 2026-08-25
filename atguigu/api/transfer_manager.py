"""
转人工会话管理器（TransferManager）

每个用户会话有三个模式（一个小型状态机）：

    machine（机器人）──申请转人工──> queue（排队）──坐席接入──> human（人工）
        ^                             |                            |
        |                        取消排队/无坐席                  坐席结束/任一方断线
        +----------------------------------------------------------------+

职责：
1. 维护每个 sender_id 当前处于哪个模式、人工模式下绑定了哪个坐席；
2. 维护全局等待队列（先进先出 deque），并给用户推送排队位置；
3. 用户消息的路由开关：machine 模式走对话引擎，human 模式原样转发给坐席；
4. 断线清理：用户断线/坐席断线时，把受影响的会话安全地"退回"机器人模式。

面试话术：转人工的本质是"消息路由的切换"——同一个 ws 通道上，
服务端根据会话当前模式决定：这条用户消息是喂给 LLM 引擎，还是转发给人工坐席。
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from atguigu.api.connection_manager import ConnectionManager, connection_manager

# 三种会话模式：机器 / 排队中 / 人工
TransferMode = Literal["machine", "queue", "human"]


@dataclass
class TransferSession:
    """单个用户的转人工会话状态"""
    sender_id: str
    mode: TransferMode = "machine"
    agent_id: str | None = None  # human 模式下绑定的坐席

    def reset_to_machine(self):
        self.mode = "machine"
        self.agent_id = None


@dataclass
class TransferResult:
    """转人工请求的处理结果：告知前端发生了什么（也用于面试讲清分支）"""
    result: Literal["queued", "no_agent_online", "already_human", "already_queued"]
    queue_position: int | None = None


class TransferManager:

    def __init__(self, connection_manager: ConnectionManager):
        self._connections = connection_manager
        # sender_id -> TransferSession（纯内存；生产可落到 DB/Redis 支持重启恢复）
        self._sessions: dict[str, TransferSession] = {}
        # 等待队列：先进先出
        self._waiting_queue: deque[str] = deque()

    # ============================== 会话查询 ==============================

    def get_or_create_session(self, sender_id: str) -> TransferSession:
        """用户 ws 连接建立后先取/建会话状态（支持用户刷新页面后找回排队/人工状态）"""
        if sender_id not in self._sessions:
            self._sessions[sender_id] = TransferSession(sender_id=sender_id)
        return self._sessions[sender_id]

    def get_session(self, sender_id: str) -> TransferSession | None:
        return self._sessions.get(sender_id)

    def queue_snapshot(self) -> list[dict[str, Any]]:
        """当前排队用户列表（发给坐席工作台）"""
        return [{"sender_id": sender_id, "position": i + 1}
                for i, sender_id in enumerate(self._waiting_queue)]

    # ============================== 用户侧动作 ==============================

    async def request_transfer(self, sender_id: str) -> TransferResult:
        """用户申请转人工"""
        session = self.get_or_create_session(sender_id)

        # 分支1：已经在人工会话中，无需重复申请
        if session.mode == "human":
            return TransferResult(result="already_human", queue_position=None)

        # 分支2：已经在排队中，直接返回最新排队位置
        if session.mode == "queue":
            position = self._queue_position(sender_id)
            await self._connections.send_to_user(sender_id, {"type": "queue_position", "position": position})
            return TransferResult(result="already_queued", queue_position=position)

        # 分支3：一个坐席都不在线，明确告知（体验上比让用户干等强）
        if not self._connections.has_online_agents():
            return TransferResult(result="no_agent_online", queue_position=None)

        # 分支4：正常入队
        session.mode = "queue"
        self._waiting_queue.append(sender_id)
        position = self._queue_position(sender_id)

        await self._connections.send_to_user(sender_id, {"type": "queue_position", "position": position})
        # 广播给所有坐席工作台：来了新的排队用户
        await self._connections.broadcast_to_agents({
            "type": "queue_update",
            "queue": self.queue_snapshot()
        })
        return TransferResult(result="queued", queue_position=position)

    async def cancel_transfer(self, sender_id: str):
        """用户在排队期间取消（一旦坐席已接入则不可取消，需走结束会话）"""
        session = self.get_session(sender_id)
        if session is None or session.mode != "queue":
            return
        self._remove_from_queue(sender_id)
        session.reset_to_machine()
        await self._connections.send_to_user(sender_id, {"type": "transfer_cancelled"})
        await self._connections.broadcast_to_agents({"type": "queue_update", "queue": self.queue_snapshot()})

    async def route_user_message(self, sender_id: str, payload: dict[str, Any]) -> bool:
        """
        消息路由（转人工的核心！）：
        - human 模式：把用户消息转发给绑定的坐席，返回 True（上层不要再走机器人引擎）
        - machine/queue 模式：返回 False，上层交给对话引擎（排队期间用户仍可与机器人聊）
        """
        session = self.get_session(sender_id)
        if session is None or session.mode != "human":
            return False

        delivered = await self._connections.send_to_agent(session.agent_id, {
            "type": "user_message",
            "sender_id": sender_id,
            "text": payload.get("text"),
            "object": payload.get("object"),
        })
        if not delivered:
            # 坐席连接已失效，兜底：退回机器人模式并告知用户
            await self.close_session(sender_id, reason="agent_lost")
        return True

    # ============================== 坐席侧动作 ==============================

    async def agent_accept(self, agent_id: str, sender_id: str) -> bool:
        """坐席从队列中接入一个用户"""
        session = self.get_session(sender_id)

        # 无效请求：该用户不在排队（已被别的坐席捷足先登 / 已取消）
        if session is None or sender_id not in self._waiting_queue:
            return False

        self._remove_from_queue(sender_id)
        session.mode = "human"
        session.agent_id = agent_id

        await self._connections.send_to_user(sender_id, {
            "type": "transfer_accepted",
            "agent_id": agent_id
        })
        await self._connections.broadcast_to_agents({"type": "queue_update", "queue": self.queue_snapshot()})
        return True

    async def agent_chat(self, agent_id: str, sender_id: str, text: str):
        """坐席回复用户（服务端主动推送给用户浏览器的就是这条）"""
        await self._connections.send_to_user(sender_id, {
            "type": "agent_message",
            "agent_id": agent_id,
            "text": text
        })

    async def close_session(self, sender_id: str, reason: Literal["agent_closed", "user_closed", "agent_lost", "user_lost"]):
        """结束人工会话，双方退回机器人模式"""
        session = self.get_session(sender_id)
        if session is None or session.mode != "human":
            return

        agent_id = session.agent_id
        session.reset_to_machine()

        await self._connections.send_to_user(sender_id, {"type": "session_closed", "reason": reason})
        if agent_id is not None:
            await self._connections.send_to_agent(agent_id, {"type": "session_closed", "sender_id": sender_id, "reason": reason})

    # ============================== 断线清理（可靠性关键路径） ==============================

    async def handle_user_disconnect(self, sender_id: str):
        """用户 ws 断开：排队中→出队；人工中→通知坐席并回收会话"""
        session = self.get_session(sender_id)
        if session is None:
            return

        if session.mode == "queue":
            self._remove_from_queue(sender_id)
            session.reset_to_machine()
            await self._connections.broadcast_to_agents({"type": "queue_update", "queue": self.queue_snapshot()})
        elif session.mode == "human":
            agent_id = session.agent_id
            session.reset_to_machine()
            if agent_id is not None:
                await self._connections.send_to_agent(agent_id, {
                    "type": "session_closed", "sender_id": sender_id, "reason": "user_lost"
                })

    async def handle_agent_disconnect(self, agent_id: str):
        """坐席 ws 断开：其名下所有人工会话全部退回机器人模式并告知用户"""
        for session in list(self._sessions.values()):
            if session.mode == "human" and session.agent_id == agent_id:
                sender_id = session.sender_id
                session.reset_to_machine()
                await self._connections.send_to_user(sender_id, {
                    "type": "session_closed", "reason": "agent_lost"
                })
        await self._connections.broadcast_to_agents({"type": "queue_update", "queue": self.queue_snapshot()})

    # ============================== 内部工具 ==============================

    def _queue_position(self, sender_id: str) -> int | None:
        try:
            return list(self._waiting_queue).index(sender_id) + 1
        except ValueError:
            return None

    def _remove_from_queue(self, sender_id: str):
        try:
            self._waiting_queue.remove(sender_id)
        except ValueError:
            pass  # 本就不在队列中，幂等处理


# 模块级单例：与 connection_manager 共享同一套连接注册表
transfer_manager = TransferManager(connection_manager)
