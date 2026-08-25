from contextlib import asynccontextmanager

from fastapi import FastAPI
from atguigu.api import chat_router, ws_router
from atguigu.api.dependencies import set_dialogue_app
from atguigu.infrastructure.client import init_http_client, dispose_http_client


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    应用生命周期：启动时装配图应用（图编译一次 + MySQL checkpointer 连接池），关闭时释放。
    checkpointer 的 async with 横跨 yield：连接池与应用同生共死。
    """
    print("应用启动期间回调到...")

    # 1. 中台 HTTP 客户端
    init_http_client()

    # 2. 对话图应用：checkpointer 生命周期由本上下文管理
    from langgraph.checkpoint.mysql.aio import AIOMySQLSaver
    from atguigu.graph.builder import build_dialogue_app, mysql_dsn_from_settings, build_checkpoint_serde

    async with AIOMySQLSaver.from_conn_string(mysql_dsn_from_settings(),
                                              serde=build_checkpoint_serde()) as checkpointer:
        await checkpointer.setup()  # 首次自动建 checkpoints 相关表
        set_dialogue_app(build_dialogue_app(checkpointer))

        yield  # 应用运行期间

        print("应用关闭回调到...")

    # 连接池已随 async with 关闭，清空全局引用
    import atguigu.api.dependencies as deps
    deps._dialogue_app = None


app = FastAPI(lifespan=lifespan)

app.include_router(chat_router.router)
app.include_router(ws_router.router)
