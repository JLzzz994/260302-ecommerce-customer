from contextlib import asynccontextmanager

from fastapi import FastAPI

from atguigu.api import chat_router, ws_router
from atguigu.api.dependencies import set_dialogue_app
from atguigu.infrastructure.client import dispose_http_client, init_http_client


@asynccontextmanager
async def lifespan(_: FastAPI):
    """装配应用级资源：业务 HTTP 客户端 + PostgreSQL LangGraph checkpointer。"""
    print("应用启动期间回调到...")

    init_http_client()

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from atguigu.graph.builder import (
        build_checkpoint_serde,
        build_dialogue_app,
        postgres_checkpoint_dsn_from_settings,
    )

    async with AsyncPostgresSaver.from_conn_string(
        postgres_checkpoint_dsn_from_settings(),
        serde=build_checkpoint_serde(),
    ) as checkpointer:
        # 首次创建 checkpoint 表；后续启动会按迁移版本检查并保持幂等。
        await checkpointer.setup()
        set_dialogue_app(build_dialogue_app(checkpointer))

        yield

        print("应用关闭回调到...")

    import atguigu.api.dependencies as deps

    deps._dialogue_app = None
    await dispose_http_client()


app = FastAPI(lifespan=lifespan)

app.include_router(chat_router.router)
app.include_router(ws_router.router)
