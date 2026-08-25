from atguigu.graph.context import TurnContext
from atguigu.task.action.base import Action, ActionResult


class ActionListener(Action):
    """占位实现：LangGraph 版中"停下等用户"由 interrupt() 承担，此 Action 仅保留注册表兼容"""
    name = "action_listen"

    async def run(self, action_kwargs, ctx: TurnContext) -> ActionResult:
        return ActionResult()
