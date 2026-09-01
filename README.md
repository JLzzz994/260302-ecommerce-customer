# 旺店通电商智能客服系统

当前业务适配分支：`feature/wangdiantong-customer-service`

该分支基于 `260302-ecommerce-customer` 的 LangGraph 内核，面向旺店通/慧策智能零售 SaaS 的商家客服场景进行业务适配。LLM 负责理解与规划，确定性 Validator、YAML Flow、Action 和人工审核负责业务执行边界。

## 1. 核心架构

```text
用户消息
  ↓
ingest
  ↓
TurnPlanner
  │  LoRA-SFT Qwen2.5-7B-Instruct
  │  vLLM OpenAI Compatible API
  ↓
TurnPlanValidator
  ├─ task      → YAML Flow → FlowCompiler → LangGraph 子图
  ├─ knowledge → 商品/订单/平台规则知识
  ├─ chitchat
  └─ clarify
          ↓
PostgreSQL Checkpoint
thread_id = sender_id
```

主图节点：

- `ingest`
- `plan`
- `validate`
- `task`
- `knowledge`
- `chitchat`
- `clarify`
- `object_dispatch`
- `pending_follow_up`
- 每个业务 Flow 编译出的 `flow_<flow_id>` 子图

## 2. 旺店通业务能力

当前只落地简历中已经使用的客服能力，不额外加入二期工作流：

1. **多平台订单状态查询**
   - 淘宝/天猫、京东、拼多多、抖音等渠道订单统一从旺店通业务中台查询。
2. **订单物流查询**
   - 查询物流公司、物流单号与最新履约进度。
3. **售后/退款建议**
   - 收集订单号和售后原因。
   - 根据订单状态生成只读建议和风险等级。
   - 不直接退款、不取消订单、不修改审批状态。
4. **相似商品推荐**
   - 调用 `/products/{product_id}/similar` 获取候选商品。
   - 中台无结果时返回可解释降级，不编造商品。
5. **人工客服转接**
   - 文本意图进入 `human_handoff` Flow。
   - `action_request_human_handoff` 调用 `TransferManager.request_transfer()`。
   - WebSocket 状态：`machine → queue → human`。

## 3. 多轮与状态持久化

缺槽由 LangGraph `interrupt()` 暂停业务子图：

```text
collect(order_number)
    ↓ 缺失
interrupt(question)
    ↓
PostgreSQL checkpoint
    ↓ 用户补订单号
Command(resume=...)
    ↓
继续原子图
```

恢复消息会同时写回：

- `user_message`
- `messages`
- 卡片场景下的 `focused_object`

历史消息使用唯一 `message_id` 标记当前轮，避免旧实现中把历史用户消息全部过滤掉。

## 4. PostgreSQL Checkpoint

```env
CHECKPOINT_DATABASE_URL=postgresql://user:password@host:5432/customer_service_checkpoint
```

使用：

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
```

应用启动时：

```python
await checkpointer.setup()
```

Checkpoint 只保存 LangGraph 运行状态、interrupt 恢复点、消息历史等，不保存旺店通订单/商品主数据。

## 5. TurnPlan 安全边界

模型只允许输出：

- `start_flow`
- `resume_flow`
- `cancel_flow`
- `set_slots`

`TurnPlanValidator` 当前确定性校验：

- 只能命中允许的轨道；
- 禁止模型启动 `system_*` 内部 Flow；
- 检查 start/resume Flow 是否存在；
- 检查 command 类型；
- 检查同轮多个业务 Flow；
- 检查 `set_slots` key 必须属于目标 Flow；
- 检查 knowledge intent 白名单；
- 检查需要订单/商品卡片的知识意图；
- 非法结果转 `ClarifyReason`，避免直接进入 Action。

高风险业务写操作不在 TurnPlanner command 白名单中。

## 6. 200 条锁定评测集

当前分支新增：

```text
data/turnplan/wangdiantong_locked_eval_200.jsonl
```

共 **200 条 / 27 个场景组**，覆盖：

- start flow
- 同轮 flow + slot
- active flow 中补 slot
- resume / cancel
- 新任务打断旧任务
- 多任务 / 多轨道
- 商品/订单卡片知识咨询
- 平台规则咨询
- chitchat
- 模糊请求

锁定集只保存 case spec，不固化完整 prompt。评测时根据当前 Flow、Intent 和 Jinja2 模板动态渲染，避免业务修改后仍然跑旧 prompt。

只验证锁定集：

```bash
uv run python -m atguigu.test.eval_turnplan_model --render-only
```

模型评测：

```bash
uv run python -m atguigu.test.eval_turnplan_model
```

输出三个核心指标：

- 严格格式合规率
- 语义精确一致率
- Validator 行为一致率

> 旧 `eval_set.jsonl` 的 100 条实验和旧 Qwen3 指标属于原实验版本，不作为当前旺店通业务分支的重新评测结果。

## 7. 无 LLM 回归

```bash
uv run python -m atguigu.test.langgraph_full_check
```

覆盖：

- 主图 / 子图编译
- interrupt 多轮收集
- interrupt 恢复答案进入历史
- 卡片直填
- checkpoint 跨实例恢复
- Validator flow/slot/intent 白名单
- 售后只读边界
- 推荐中台失败降级
- 无在线坐席时人工转接降级
- knowledge / chitchat / clarify

## 8. 业务中台接口约定

```text
GET /orders/{order_id}
GET /orders/{order_id}/logistics
GET /products/{product_id}
GET /products/{product_id}/similar?limit=3
```

订单、商品、履约数据必须从中台查询，LLM 不允许编造业务事实。

## 10. Vue 3 演示前端

当前分支新增 `frontend/`，不是旧仓库页面复制，而是直接围绕当前 LangGraph + WebSocket 后端做的最小正式演示。

包含：

- 用户聊天端；
- 历史消息加载；
- WebSocket 智能客服；
- 转人工 / 取消排队；
- 人工坐席上线；
- 排队列表；
- 坐席接入；
- 人工消息；
- 结束人工会话。

启动：

```bash
cd frontend
npm install
npm run dev
```

Vite 默认把：

```text
/api → http://127.0.0.1:18082
/ws  → ws://127.0.0.1:18082
```

代理到 FastAPI 后端。

坐席端的 `agent_chat` / `close_session` 会校验 `session.agent_id`，不能跨坐席操作其他人的人工会话。

## 11. 旧实验文档

`docs/turnplanner-*.md`、`docs/turnplan-dataset-log.md` 中部分内容记录原 Qwen3-1.7B / 100 条 eval 实验，用于保留实验过程和排障记录。

当前业务版本请优先阅读：

```text
README.md
docs/wangdiantong-interview-story.md
```
