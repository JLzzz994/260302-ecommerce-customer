from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE_PATH = PROJECT_DIR / ".env"


class Settings(BaseSettings):
    """应用配置，完整示例见项目根目录 .env.example。"""

    # 表达层大模型
    llm_model: str
    llm_base_url: str
    llm_api_key: str

    # TurnPlanner 路由模型：LoRA-SFT 后通过 vLLM OpenAI 兼容接口部署
    llm_router_model: str = "turnplanner"
    llm_router_base_url: str
    llm_router_api_key: str

    # 旺店通业务中台 API
    commerce_api_base_url: str

    # LangGraph Checkpoint 专用 PostgreSQL。
    # 与订单/商品等业务数据源解耦，thread_id=sender_id。
    checkpoint_database_url: str

    app_host: str = "0.0.0.0"
    app_port: int = 18082

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()  # type: ignore


if __name__ == "__main__":
    print(settings.llm_base_url)
