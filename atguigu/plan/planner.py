import json
from dataclasses import asdict
from typing import Any
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from atguigu.graph.context import TurnContext
from atguigu.knowledge.intents import KnowledgeIntent
from atguigu.prompt.loader import load_prompt_template
from atguigu.infrastructure.llm import router_client
from atguigu.task.flows.flows import FlowsList
from atguigu.plan.turn_plan import TurnPlan


class TurnPlanner:
    """
    路由规划器：调用（微调后的）路由模型生成本轮 TurnPlan
    LangGraph 版：从 TurnContext + 图状态里的流程运行时构建提示词输入
    """

    async def predict(self,
                      ctx: TurnContext,
                      flows_list: FlowsList,
                      knowledge_intents: dict[str, KnowledgeIntent],
                      *,
                      active_flow: str | None = None,
                      active_flow_step: str | None = None,
                      slots: dict[str, Any] | None = None,
                      paused_flows: dict[str, dict] | None = None,
                      ) -> TurnPlan:
        # 1. 准备提示词模版中的变量值
        prompt_inputs: dict[str, Any] = self._build_prompt_inputs(
            ctx, flows_list, knowledge_intents,
            active_flow=active_flow, active_flow_step=active_flow_step,
            slots=slots, paused_flows=paused_flows,
        )

        # 2. 格式化模板以及调用LLM
        return await self._invoke(prompt_inputs)

    def _build_prompt_inputs(self,
                             ctx: TurnContext,
                             flows_list: FlowsList,
                             knowledge_intents: dict[str, KnowledgeIntent],
                             *,
                             active_flow: str | None,
                             active_flow_step: str | None,
                             slots: dict[str, Any] | None,
                             paused_flows: dict[str, dict] | None,
                             ) -> dict[str, Any]:
        # 1. 会话相关（本轮消息 + 之前的对话历史）
        user_message_str = ctx.user_message_text()
        current_conversation_str = ctx.history_text(last_n=10)

        # 2. 任务相关（图状态里的流程运行时）
        active_task_json_str = json.dumps({
            "flow_id": active_flow,
            "step_id": active_flow_step,
            "slots": slots or {},
        }, ensure_ascii=False) if active_flow is not None else "null"
        interrupted_tasks_json_str = json.dumps(
            [{"flow_id": flow_id, "slots": flow_slots} for flow_id, flow_slots in (paused_flows or {}).items()],
            ensure_ascii=False)

        # 3. 卡片相关
        focused_object_json_str = json.dumps(ctx.focused_object,
                                             ensure_ascii=False) if ctx.focused_object is not None else "null"

        # 4. 清单相关
        available_flows_json_str = json.dumps({
            "flows": [
                {
                    k: v for k, v in asdict(flow_obj).items() if k != "steps"
                } for flow_obj in flows_list.flows if not flow_obj.flow_id.startswith("system_")
            ]
        }, ensure_ascii=False)
        knowledge_intents_json_str = json.dumps([
            {"id": intent.id, "description": intent.description} for intent in knowledge_intents.values()
        ], ensure_ascii=False)

        return {
            "user_message": user_message_str,
            "current_conversation": current_conversation_str,
            "active_task_json": active_task_json_str,
            "interrupted_tasks_json": interrupted_tasks_json_str,
            "focused_object_json": focused_object_json_str,
            "available_flows_json": available_flows_json_str,
            "knowledge_intents_json": knowledge_intents_json_str,
        }

    async def _invoke(self, prompt_inputs: dict[str, Any]) -> TurnPlan:
        # 1. 获取提示词模版中的内容
        prompt_template_str = load_prompt_template("turn_plan")

        # 2. 格式化提示词模版中的变量
        prompt_template = PromptTemplate.from_template(template=prompt_template_str, template_format="jinja2")

        # 3. LCEL方式通过链来定义
        chain = prompt_template | router_client | JsonOutputParser()

        # 4. 执行链
        llm_result_dict = await chain.ainvoke(prompt_inputs)

        # 5. 返回
        return TurnPlan.from_dict(llm_result_dict)
