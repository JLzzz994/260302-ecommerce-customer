"""
统一管理所有xxxservice（LangGraph 版）

图应用是"重对象"（编译图+流程子图），但完全无并发可变状态：
- 图编译产物只读，全局单例
- MySQL checkpointer 连接池在应用 lifespan 里建/释放
所以不再用 Depends 每请求构建（旧版每次请求重建整条组件链的开销问题就此消失）
"""
from typing import Annotated
from fastapi import Depends

from atguigu.services.dialogue_service import DialogueService

# 由 app.lifespan 装配好后注入（图编译一次，全局复用）
_dialogue_app = None


def set_dialogue_app(dialogue_app):
    global _dialogue_app
    _dialogue_app = dialogue_app


def get_dialogue_app():
    assert _dialogue_app is not None, "DialogueApp 未初始化：应用启动时需调用 set_dialogue_app"
    return _dialogue_app


DialogueAppDep = Annotated[object, Depends(get_dialogue_app)]


def get_dialogue_service(app: DialogueAppDep):
    return DialogueService(app)


DialogueServiceDep = Annotated[DialogueService, Depends(get_dialogue_service)]
