# 旺店通电商智能客服系统（业务适配分支）

当前分支：`feature/wangdiantong-customer-service`

该分支基于原 `260302-ecommerce-customer` 的 LangGraph 对话内核，按旺店通多平台电商商家客服场景进行业务适配。核心目标不是让 LLM 直接操作 ERP，而是让模型负责“理解与规划”，由确定性流程和受控 Action 负责查询，高风险写操作保留人工复核。

## 核心链路

```text
用户消息
  ↓
ingest
  ↓
TurnPlanner（LoRA-SFT Qwen2.5-7B-Instruct / vLLM）
  ↓
TurnPlanValidator
  ├─ task → YAML Flow → FlowCompiler → LangGraph 子图
  ├─ knowledge → 商品/订单/规则知识
  ├─ chitchat
  └─ clarify
                ↓
      PostgreSQL Checkpoint
      thread_id = sender_id
```

## 旺店通业务能力

- 多平台订单状态查询：通过中台接口聚合淘宝/天猫、京东、拼多多、抖音等订单状态。
- 物流查询：查询物流公司、运单号和最新履约进度。
- 售后/退款建议：收集订单和售后原因，结合订单状态生成处理建议；**不直接提交退款、不取消订单、不修改审批状态**。
- 相似商品推荐：调用商品中台 `/products/{product_id}/similar` 获取候选商品。
- 人工转接：WebSocket 维护 machine → queue → human 路由状态，复杂和高风险问题切人工坐席。
- 多轮缺槽：LangGraph `interrupt()` 暂停流程，用户下一轮通过 `Command(resume=...)` 恢复。

## Checkpoint

LangGraph 运行状态独立使用 PostgreSQL：

```env
CHECKPOINT_DATABASE_URL=postgresql://user:password@host:5432/customer_service_checkpoint
```

应用启动时使用 `AsyncPostgresSaver` 并执行 `setup()` 初始化 checkpoint 表。订单、商品等业务主数据不写入该库，仍通过旺店通业务中台 API 获取。

## 安装

该分支已将依赖从 `langgraph-checkpoint-mysql` 切换为 `langgraph-checkpoint-postgres`。原 `uv.lock` 基于 MySQL 依赖，已从该分支移除。

```bash
uv sync
```

首次同步会根据新的 `pyproject.toml` 生成 PostgreSQL 版本锁文件。

## 业务接口约定

```text
GET /orders/{order_id}
GET /orders/{order_id}/logistics
GET /products/{product_id}
GET /products/{product_id}/similar?limit=3
```

接口响应约定统一从 `data` 字段读取。推荐接口可返回 `data: []` 或 `data: {"items": []}`。

## 安全边界

TurnPlanner 只能输出白名单 TurnPlan：

- `start_flow`
- `resume_flow`
- `cancel_flow`
- `set_slots`

模型不拥有退款、取消订单、修改库存、修改审批等写权限。售后链路只生成建议和风险等级，实际高风险动作由人工客服复核后在业务系统执行。
