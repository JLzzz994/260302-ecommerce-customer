"""
生产装配：真实组件 + MySQL checkpointer 的完整对话图

组件清单与架构变化：
- TurnPlanner/TurnPlanValidator/ClarifyResponder：同源复用（签名改为 TurnContext）
- KnowledgeHandler/ChitChatHandler：同源复用
- 流程：YAML → FlowCompiler 编译成子图挂进主图（替代 FlowExecutor 解释执行）
- 持久化：AIOMySQLSaver checkpointer（thread_id=sender_id），替代 DialogueRepository 整体JSON落库
- 主图装配见 graph/main_graph.py；checkpointer 生命周期由 api/app.py 的 lifespan 管理
"""

from pathlib import Path

from langgraph.checkpoint.mysql.aio import AIOMySQLSaver

from atguigu.clarify.responder import ClarifyResponder
from atguigu.chitchat.handler import ChitChatHandler
from atguigu.chitchat.responder import ChitChatResponder
from atguigu.config.config import settings
from atguigu.graph.app import DialogueApp
from atguigu.graph.main_graph import build_main_graph
from atguigu.knowledge.handler import KnowledgeHandler
from atguigu.knowledge.intents import KNOWLEDGE_INTENTS
from atguigu.knowledge.providers.knowledge import (ApiOrderProvider, ApiProductProvider,
                                                   FAQDefaultProvider, RAGDefaultProvider)
from atguigu.knowledge.providers.register import ProviderRegister
from atguigu.knowledge.responder import KnowledgeResponder
from atguigu.plan.planner import TurnPlanner
from atguigu.plan.validator import TurnPlanValidator
from atguigu.task.action.buidler import build_action_runner
from atguigu.task.flows.loader import FlowLoader

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[2]
FLOW_CONFIG_DIR = PROJECT_ROOT_DIR / "flow_config"
FLOW_CONFIGS = ["system_flows.yml", "user_flows.yml"]


def build_dialogue_app(checkpointer=None) -> DialogueApp:
    """装配生产对话应用。checkpointer 可注入（测试用内存版），默认无持久化"""
    flows_list = FlowLoader().load_multi_yaml([FLOW_CONFIG_DIR / cfg for cfg in FLOW_CONFIGS])

    graph = build_main_graph(
        planner=TurnPlanner(),
        validator=TurnPlanValidator(),
        clarify_responder=ClarifyResponder(),
        knowledge_handler=KnowledgeHandler(
            knowledge_intents=KNOWLEDGE_INTENTS,
            knowledge_responder=KnowledgeResponder(),
            providers_register=ProviderRegister(providers=[
                ApiOrderProvider(),
                ApiProductProvider(),
                FAQDefaultProvider(),
                RAGDefaultProvider()
            ])
        ),
        chitchat_handler=ChitChatHandler(chat_responder=ChitChatResponder()),
        flows_list=flows_list,
        action_runner=build_action_runner(),
        checkpointer=checkpointer,
    )

    return DialogueApp(graph, checkpointer)


def mysql_dsn_from_settings() -> str:
    """DATABASE_URL(mysql+aiomysql://...) → langgraph-checkpoint-mysql 的 mysql:// DSN"""
    from sqlalchemy.engine.url import make_url
    url = make_url(settings.database_url)
    return f"mysql://{url.username}:{url.password}@{url.host}:{url.port or 3306}/{url.database}"


# checkpoint 序列化 allowlist：图状态里的领域类型（TurnPlan/命令/枚举等）
# 注册后 msgpack 反序列化不再告警，也避免未来严格模式被拦截
def build_checkpoint_serde():
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from atguigu.plan import turn_plan, commands
    from atguigu.domain import messages

    allowlist = []
    for module in (turn_plan, commands, messages):
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and obj.__module__ == module.__name__:
                allowlist.append(obj)

    return JsonPlusSerializer().with_msgpack_allowlist(allowlist)
