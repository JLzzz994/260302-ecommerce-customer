from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# 读取.env文件
PROJECT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE_PATH = PROJECT_DIR / ".env"


class Settings(BaseSettings):
    """
    接收.env文件中的环境变量

    LLM_MODEL=qwen-plus
    LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    LLM_API_KEY=sk-REDACTED

    COMMERCE_API_BASE_URL=http://192.168.200.133:18081

    DATABASE_URL=mysql+aiomysql://root:REDACTED@192.168.200.133:3306/customer_service?charset=utf8mb4

    APP_HOST=0.0.0.0
    APP_PORT=18082
    """

    llm_model: str  # 模型名字
    llm_base_url: str  # 模型服务平台的地址
    llm_api_key: str  # 模型服务平台的api_key
    # 路由规划专用：微调后的 Qwen3-1.7B（vLLM --enable-lora 部署，取值为 LoRA 模块别名 turnplanner）
    llm_router_model: str = "turnplanner"
    # AutoDL 公网 HTTPS 端口映射（.env 里 LLM_ROUTER_BASE_URL 可覆盖）
    llm_router_base_url: str = "https://INSTANCE-REDACTED.westc.seetacloud.com:8443/v1"
    # vLLM 启动参数 --api-key 的值
    llm_router_api_key: str = "sk-REDACTED"
    commerce_api_base_url: str  # 中台服务的地址
    database_url: str  # AI应用对应的数据库地址
    app_host: str  # AI应用的访问域名
    app_port: int  # AI 应用的端口

    # 实例化SettingsConfigDict对象一定要有变量接收  并且变量的名字一定要叫model_config
    model_config = SettingsConfigDict(env_file=ENV_FILE_PATH, env_file_encoding="utf-8",
                                      extra="ignore")  # extra="ignore" 忽略掉.env文件中多余的key_value


settings = Settings()  # type: ignore

if __name__ == '__main__':
    print(settings.llm_base_url)
