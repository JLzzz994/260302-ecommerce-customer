# 旺店通电商智能客服系统｜当前分支面试叙事

> 对应分支：`feature/wangdiantong-customer-service`
>
> 本文只描述当前旺店通业务分支。原 Qwen3-1.7B / 100 条 eval 的训练与 bf16 合并事故记录仍保留在 legacy 文档中，不和当前版本混讲。

## 1. 一句话项目定位

面向旺店通多平台电商商家客服场景，建设一个“**LLM 只做 TurnPlan，确定性流程负责执行，高风险动作转人工**”的智能客服系统，覆盖订单查询、物流查询、售后/退款建议、相似商品推荐和人工转接。

## 2. 主链路怎么讲

```text
User
 ↓
ingest
 ↓
TurnPlanner
 ↓
TurnPlan
 ↓
Validator
 ├─ task → FlowCompiler → LangGraph 子图
 ├─ knowledge
 ├─ chitchat
 └─ clarify
 ↓
PostgreSQL Checkpoint
```

TurnPlanner 不直接调订单、退款、库存接口，只输出结构化计划。

## 3. 为什么 YAML Flow + FlowCompiler

订单、物流、售后等流程使用 YAML 描述：

- collect：收集订单号、售后原因等槽位；
- action：调用确定性 Action；
- condition：根据 slot 路由；
- end：结束业务流程。

`FlowCompiler` 在启动时把 YAML 编译为 LangGraph 子图，而不是运行时手写 while 状态机。

新增标准业务流程主要改 YAML + Action，不修改主图核心。

## 4. 多轮中断恢复

缺少订单号时：

```text
collect
 ↓
interrupt
 ↓
checkpoint
 ↓
用户补订单号
 ↓
Command(resume)
 ↓
原节点继续执行
```

恢复答案同时进入 `messages`，因此下一轮 Planner 能看到用户之前补充过的订单号和上下文。

## 5. 为什么 PostgreSQL Checkpoint

Checkpoint 是 Agent/Graph 运行状态，不是订单业务库。

保存：

- messages
- active_flow
- flow_step
- slots
- paused_flows
- pending_intents
- interrupt 恢复点

按 `thread_id=sender_id` 隔离用户。

项目把 Checkpoint 单独放 PostgreSQL，订单、商品、物流继续通过旺店通中台 API 获取。

## 6. Validator 为什么是项目重点

模型理解和规划允许犯错，但不能让错误计划进入执行层。

当前 Validator 防线包括：

1. 单轨道约束；
2. command 白名单；
3. user Flow 白名单，禁止启动 `system_*`；
4. resume Flow 合法性；
5. slot key 必须属于当前目标 Flow；
6. knowledge intent 白名单；
7. 商品/订单卡片依赖校验。

例如模型输出：

```json
{
  "task": {
    "commands": [
      {
        "command": "set_slots",
        "slots": {
          "refund_reason": "太贵"
        }
      }
    ]
  },
  "knowledge": null,
  "chitchat": null
}
```

如果当前是 `order_status_query`，该 Flow 只允许 `order_number`，Validator 会以 `INVALID_TASK_SLOTS` 拒绝，不让错误数据进入状态。

## 7. 售后为什么不是“自动退款”

当前 `refund_request` 实际语义已经改成“售后/退款建议”。

```text
order_number
 ↓
refund_reason
 ↓
action_build_refund_advice
 ↓
risk_level + refund_advice
 ↓
人工复核
```

模型没有：

- refund
- cancel_order
- update_order
- approve

这类 command。

因此 AI 可以解释和建议，但不能直接改变高风险业务状态。

## 8. 人工转接怎么落地

文本“我要转人工”：

```text
TurnPlanner
 ↓
start_flow(human_handoff)
 ↓
action_request_human_handoff
 ↓
TransferManager.request_transfer
 ↓
machine → queue → human
```

WebSocket 用户端和坐席端分别维护长连接。

人工接入后，用户消息不再进入 LLM，而是直接路由给绑定坐席。

## 9. TurnPlanner 微调口径

当前业务分支按简历技术口径：

- Qwen2.5-7B-Instruct
- LoRA-SFT
- vLLM
- OpenAI Compatible API
- TurnPlanner 与表达层大模型分离

目的不是给模型注入知识，而是把“意图 → flow / command / slot”的稳定结构化映射内化到路由模型。

## 10. 200 条锁定集

当前分支已经提交：

`data/turnplan/wangdiantong_locked_eval_200.jsonl`

共 200 条 / 27 个业务组。

与旧版不同，它不保存固定的完整 prompt；每次评测时按当前：

- YAML Flow
- knowledge intents
- Jinja2 prompt

动态重新渲染。

先执行：

```bash
uv run python -m atguigu.test.eval_turnplan_model --render-only
```

确认锁定集和当前代码契约一致，再调用 vLLM 正式测：

```bash
uv run python -m atguigu.test.eval_turnplan_model
```

## 11. 关于简历中的结果数字

简历当前记录的历史项目结果为：

- 200 条锁定测试集；
- 格式合规率 93.50% → 99.50%；
- 语义一致率 60.43% → 97.99%。

这些数字属于项目历史结果口径。

**当前 GitHub 旺店通分支已经把 200 条锁定集、评测脚本和 Qwen2.5-7B 路由接口对齐，但在没有重新连接对应 vLLM 权重跑完整评测前，不把旧 100 条 Qwen3 实验结果冒充成当前分支复现实验。**

面试时如果展示仓库，可以说：

> 简历数字是项目当时锁定集的结果；这个公开分支重新整理了业务和评测契约，所以我把锁定 case 固化成 200 条并改成动态渲染。连接对应路由模型后，用同一个脚本可以复现当前分支指标，代码里没有硬编码成绩。

## 12. 最容易被追问的三个点

### 为什么不用大模型直接 Tool Calling？

因为客服业务存在高风险动作和严格流程约束。TurnPlanner 只负责规划，Validator + Flow + Action 决定能不能执行，可以做到权限最小化和错误可控。

### 为什么不是每个流程都做一个 Agent？

这里订单查询、物流查询、售后建议更适合确定性工作流，不需要让多个自由 Agent 互相聊天。LangGraph 的价值主要是状态、子图、中断恢复和条件路由。

### PostgreSQL Checkpoint 和业务数据库什么关系？

没有必要是同一个库。Checkpoint 是 Graph runtime state；订单、商品、物流是旺店通业务主数据。分库后职责更清楚，也避免 Agent 状态表侵入 ERP/WMS 核心库。
