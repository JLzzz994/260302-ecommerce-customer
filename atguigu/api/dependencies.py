"""API 依赖装配（LangGraph 版）。

图应用是重对象，但编译产物只读，可作为应用级单例复用。
PostgreSQL checkpointer 的连接生命周期由 FastAPI lifespan 管理，
每个用户通过 thread_id=sender_id 隔离自己的 GraphState。
"""

from typing import Annotated

from fastapi import Depends

from atguigu.services.dialogue_service import DialogueService


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
