from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# 读取.env文件
PROJECT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE_PATH = PROJECT_DIR / ".env"


class Settings(BaseSettings):
    """
    接收.env文件中的环境变量（完整模板见项目根目录 .env.example）

    LLM_MODEL=deepseek-ai/DeepSeek-V4-Flash
    LLM_BASE_URL=https://api.siliconflow.cn/v1
    LLM_API_KEY=<你的大模型平台 API Key>

    LLM_ROUTER_MODEL=turnplanner
    LLM_ROUTER_BASE_URL=<vLLM 部署地址，如 https://<实例映射域名>:8443/v1>
    LLM_ROUTER_API_KEY=<vLLM 启动参数 --api-key 的值>

    COMMERCE_API_BASE_URL=<中台服务地址>

    DATABASE_URL=mysql+aiomysql://<user>:<password>@<host>:3306/customer_service?charset=utf8mb4

    APP_HOST=0.0.0.0
    APP_PORT=18082
    """

    llm_model: str  # 模型名字
    llm_base_url: str  # 模型服务平台的地址
    llm_api_key: str  # 模型服务平台的api_key
    # 路由规划专用：微调后的 Qwen3-1.7B（vLLM --enable-lora 部署，取值为 LoRA 模块别名 turnplanner）
    llm_router_model: str = "turnplanner"
    # vLLM 部署的公网地址（AutoDL 端口映射，实例重启会变），在 .env 中配置
    llm_router_base_url: str
    # vLLM 启动参数 --api-key 的值，在 .env 中配置
    llm_router_api_key: str
    commerce_api_base_url: str  # 中台服务的地址
    database_url: str  # AI应用对应的数据库地址
    app_host: str  # AI应用的访问域名
    app_port: int  # AI 应用的端口
    # 对话引擎实现开关：langgraph=LangGraph编排版（默认） / classic=自研状态机版
    # 两个实现组件同源、行为对齐，用于A/B对比与回退
    dialogue_engine: str = "langgraph"

    # 实例化SettingsConfigDict对象一定要有变量接收  并且变量的名字一定要叫model_config
    model_config = SettingsConfigDict(env_file=ENV_FILE_PATH, env_file_encoding="utf-8",
                                      extra="ignore")  # extra="ignore" 忽略掉.env文件中多余的key_value


settings = Settings()  # type: ignore

if __name__ == '__main__':
    print(settings.llm_base_url)
