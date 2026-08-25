"""
TurnContext：传给 Action / Provider / Responder 的轻量读视图

替代被删除的 DialogueState：组件们只关心"当前用户消息、最近历史、槽位、卡片、过场变量"，
不再需要会话/轮次/任务上下文那些手工状态机概念。
"""

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from atguigu.domain.messages import UserMessage


def message_text(message: AnyMessage) -> str:
    """消息文本；卡片消息渲染成【id=.. label=..】格式（给提示词用）"""
    content = message.content if isinstance(message.content, str) else str(message.content)
    obj = message.additional_kwargs.get("object")
    if obj is not None:
        label = "订单" if obj.get("type") == "order" else "商品"
        attrs = "|".join(f"{k}={v}" for k, v in (obj.get("attributes") or {}).items())
        return f"【id={obj.get('id')} label={label} title={obj.get('title')} attributes={attrs}】"
    return content.strip()


def build_history_text(messages: list[AnyMessage], last_n: int = 10) -> str:
    """构建提示词里的对话历史：USER:xxx / BOT:xxx（不含本轮输入）"""
    lines = []
    for message in messages[-last_n:]:
        role = "USER" if isinstance(message, HumanMessage) else "BOT"
        lines.append(f"{role}: {message_text(message)}")
    return "\n".join(lines)


@dataclass(slots=True)
class TurnContext:
    """每个需要读对话上下文的组件（Action/Provider/Responder）都接收它"""
    user_message: UserMessage | None = None          # 本轮用户消息
    history_messages: list[AnyMessage] = field(default_factory=list)  # 本轮之前的历史
    slots: dict[str, Any] = field(default_factory=dict)               # 当前流程槽位
    flow_context: dict[str, Any] | None = None       # 过场话术渲染变量
    focused_object: dict[str, Any] | None = None     # 用户点击的卡片

    def user_message_text(self) -> str:
        """本轮用户消息的提示词表示（文本或卡片渲染）"""
        if self.user_message is None:
            return ""
        obj = self.user_message.object
        if obj is not None:
            label = "订单" if obj.type == "order" else "商品"
            attrs = "|".join(f"{k}={v}" for k, v in (obj.attributes or {}).items())
            return f"【id={obj.id} label={label} title={obj.title} attributes={attrs}】"
        return (self.user_message.text or "").strip()

    def history_text(self, last_n: int = 10) -> str:
        return build_history_text(self.history_messages, last_n)
