"""LLM 客户端：生成模型与 TurnPlanner 路由模型分层部署。"""

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from atguigu.config.config import settings


def _build_client(model: str, base_url: str, api_key: str, **kwargs: Any) -> BaseChatModel:
    return init_chat_model(
        model=model,
        model_provider="openai",
        base_url=base_url,
        api_key=api_key,
        temperature=0,
        timeout=60,
        **kwargs,
    )


# 表达层：澄清、知识问答、闲聊和业务话术生成，保留能力更强的通用模型。
llm_client: BaseChatModel = _build_client(
    settings.llm_model,
    settings.llm_base_url,
    settings.llm_api_key,
)

# 路由层：TurnPlanner 专用。
# 旺店通业务分支按简历口径使用 LoRA-SFT 后的 Qwen2.5-7B-Instruct，
# 通过 vLLM 暴露 OpenAI 兼容接口；代码只依赖接口协议，不绑死具体模型路径。
router_client: BaseChatModel = _build_client(
    settings.llm_router_model,
    settings.llm_router_base_url,
    settings.llm_router_api_key,
)


if __name__ == "__main__":
    print(llm_client.invoke("你好"))
