"""
定义LLM 客户端
标准写法（PEP8规范）
# 1. sdk自带的依赖包

# 2. 第三组件的依赖包

# 3. 自己应用的依赖包
"""
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
        temperature=0,  # 尽量保证同样的输入得到同样的输出（稳定性）
        timeout=60,
        **kwargs,
    )


# 表达层（澄清润色/知识问答/闲聊/话术生成）：继续走大模型 API
llm_client: BaseChatModel = _build_client(settings.llm_model, settings.llm_base_url, settings.llm_api_key)

# 路由层（TurnPlanner 专用）：微调后的 Qwen3-1.7B，vLLM --enable-lora 部署的 turnplanner
# 铁律：训练时按非思考模式渲染，推理必须同样传 enable_thinking=False，与训练分布对齐
router_client: BaseChatModel = _build_client(
    settings.llm_router_model,
    settings.llm_router_base_url,
    settings.llm_router_api_key,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

if __name__ == '__main__':
    invoke = llm_client.invoke('你好')
    print(invoke)
