> [!WARNING]
> **Legacy 实验文档**：本文记录原仓库的 Qwen3-1.7B / 100 条 eval 实验过程，不代表 `feature/wangdiantong-customer-service` 当前业务版本。
> 当前分支使用 Qwen2.5-7B-Instruct 路由口径、PostgreSQL Checkpoint 和 200 条动态渲染锁定集。
> 当前版本请优先阅读 `README.md` 与 `docs/wangdiantong-interview-story.md`。

# TurnPlanner 模型微调方案与面试话术

> 目标：把系统里最高频、最该确定性的 LLM 调用点——路由规划器（TurnPlanner）——从「调 qwen-plus 大模型 API」换成「微调后的本地 Qwen3-1.7B 小模型」，在**保留三层防御（规划→校验→澄清）不变**的前提下，降成本、降延迟、降澄清率。
>
> 红线：**微调只动「理解」，不动「裁决」**。`TurnPlanValidator` 那层确定性校验一行不改。
>
> ---
> **✅ 落地结果（2026-08-20 已闭环）**：格式合规 94%→**100%**、语义一致 71%→**99%**，双双反超大模型老师；路由延迟秒级→几十毫秒。
> 中间经历 bf16 合并事故（merged 权重微调信号被量化抹除 52.8%，评测 40%/0%），根因与修复见 `turnplanner-bf16-merge-pitfall.md`；现行部署为 **vLLM `--enable-lora` 运行时挂载 adapter**（非合并权重），指南见 `turnplanner-deploy-autodl.md`；面试叙事见 `turnplanner-interview-story.md`。

---

## 1. 为什么要微调 TurnPlanner（选点理由）

### 1.1 项目里的 LLM 触点全景

全项目 5 个 LLM 调用点，全部经由 `atguigu/infrastructure/llm.py` 里的全局 `llm_client`（temperature=0，OpenAI 兼容协议）：

| # | 调用点 | 任务性质 | 微调适配度 |
|---|---|---|---|
| 1 | `atguigu/plan/planner.py`（TurnPlanner） | 意图分类 + 槽位抽取 + 命令生成（结构化 NLU） | ⭐⭐⭐ 最高 |
| 2 | `atguigu/clarify/responder.py` | 澄清话术润色（受限改写） | ⭐⭐ 可被规则替代 |
| 3 | `atguigu/task/action/builtin/response.py`（rephrase/generate） | 话术改写/生成 | ⭐⭐ 中 |
| 4 | `atguigu/knowledge/responder.py` | RAG 生成 | ⭐ 低（数据分布漂移） |
| 5 | `atguigu/chitchat/responder.py` | 开放式闲聊生成 | ✗ 不建议 |

### 1.2 为什么只微调 TurnPlanner

TurnPlanner 满足微调的三个硬条件：

1. **高频**——每条文本消息必走一次，是最高频调用；
2. **结构化输出**——输出是固定 schema 的 `TurnPlan` JSON，监督信号天然可得；
3. **有确定性兜底可当免费质检器**——`TurnPlanValidator` 可直接复用为蒸馏数据的质检闸门（通过→进候选池，被拒→人工分诊后修复）。注意它只能当**质检器**、当不了**标注器**：它查不出「flow 选错但存在」「槽位值幻觉」这类 valid-but-wrong 错误，这类样本必须人工把关。

其余 4 个点是「表达层/生成层」，数据分布漂移、开放式生成，微调性价比低，继续用大模型。

### 1.3 微调覆盖哪些能力点（对应 4.3.1~4.3.5）

| 能力点 | TurnPlanner 是否命中 | 说明 |
|---|---|---|
| 4.3.1 意图识别 | ✅ 完整 | 三轨道 `task/knowledge/chitchat` + `start_flow.flow` 选 flow_id |
| 4.3.2 槽位抽取 | ⚠️ 部分 | `set_slots.slots` 的抽取在模型；缺失/冲突判断在 `FlowExecutor` 的 collect 步骤 |
| 4.3.3 澄清生成 | ❌ 不命中 | 澄清话术是 `ClarifyResponder` 另一个调用点 |
| 4.3.4 流程状态管理 | ⚠️ 部分 | `resume_flow`/`cancel_flow` 命令**决策**在模型；状态**执行**在 `CommandProcessor`/`FlowExecutor` |
| 4.3.5 异常/人工升级 | ❌ 基本不命中 | 是 `system_` 兜底流程 + Action 的确定性逻辑 |

> 结论：微调 TurnPlanner 完整覆盖 4.3.1 + 4.3.2（抽取层），部分覆盖 4.3.4（决策层）。想覆盖 4.3.3 要另微调 ClarifyResponder；4.3.5 是工程侧（改 YAML/Action），不是微调能解决的。

---

## 2. 基座选型理由

### 2.1 为什么选 Qwen3-1.7B

看四点（按说服力排序）：

1. **延迟与成本（第一论据）**——TurnPlanner 在每条文本消息的关键路径上，输入 2~3K token、输出只有几十 token，路由延迟几乎全是大模型 API 的 prefill 时间。1.7B 本地部署把路由延迟压到几十毫秒、边际成本趋近于零，这是继续调 API 永远给不了的；
2. **任务性质**——意图识别 + 槽位抽取是 NLU 分类/抽取任务，不是开放式生成，1B~3B 是甜点区。但 1.7B 是该区间的**下沿**：分类不难，难在「读长状态做条件决策」（resume/start 区分）和边角 case 的指令遵循——所以 **Qwen3-4B（dense）列为备胎**，评测里 resume/start 混淆率或槽位幻觉压不住就直接升级，不恋战；
3. **语种**——业务是中文电商客服（订单号/退款/物流槽位），Qwen 系列中文实体抽取最强；且 Qwen3 全系共享 chat template，qwen-plus 蒸馏数据格式可无损迁移；
4. **迁移成本的正确算法**——不是「同家族行为差异小」（微调后行为由 SFT 数据分布主导，与是不是亲戚无关）。真实成本在三处：① 蒸馏老师就是 qwen-plus，数据天然对齐；② validator 是纯确定性 schema 校验，换任何模型都一行不改；③ 会变的是各 `ClarifyReason` 的**触发频率分布**（小模型多轨道误判、槽位漏抽更多，澄清率必然漂移）——所以要重跑澄清率基线，而不是重调 validator。

起步选 `Qwen3-1.7B`（注意命名：Qwen3 系列没有 `-Instruct` 后缀，`Qwen3-1.7B` 本身就是指令微调后的对话版，`-Base` 才是基座；不要写成 `Qwen3-1.7B-Instruct`，仓库不存在会 404）。**Qwen3-1.7B 是混合思考模型：训练数据必须以 `enable_thinking=False` 渲染，否则上线后会间歇性吐思考链，`JsonOutputParser` 直接炸。**

### 2.2 为什么不用 Qwen3.5（4B/9B）

Qwen3.5 小模型跑分确实强（4B 的 OCRBench 85.0 甚至高于一些 30B 级模型），但**升级方向与 TurnPlanner 的负载错位——能力买不进来，架构代价却要全额支付**：

- **升级的东西用不上**：核心变化是 Hybrid 架构（Gated DeltaNet + Attention 交替）+ 统一视觉-语言能力，主打长文本、代码、多模态 Agent。TurnPlanner 输入撑死几 K token、纯文本 NLU、输出几十 token JSON——跑分涨的维度一项都命中不了；
- **代价全踩在关键路径上**：Hybrid 架构 prefill 更复杂，社区实测 TTFT 明显慢于 Qwen3 dense，而路由延迟的大头就是 TTFT；Mamba 类缓存的 prefix caching 在 vLLM 仍是实验特性（TurnPlanner「长而稳定的模板头 + 短而多变的尾」恰是 prefix cache 教科书场景，dense 上是默认成熟特性）；4B 实测 16GB 卡 vLLM 部署 OOM、要求 transformers ≥ 4.57，guided JSON 支持未验证；
- **9B 另加一条**：BF16 权重 ~18GB，微调/部署门槛跳一档，撑不起「高频、便宜、低延迟」的主线。

> Qwen3.5 的 VL 能力留给后续图片需求：用户传订单截图 → VL 模型前置提取字段/映射 `FocusedObject` → 再进现有文本管线，与 TurnPlanner 解耦，互不影响。

选型原则：**按「能力与负载对齐 + 成本收益」选型，不是越大越新越好；先用 1.7B 跑出基线，数据证明不够再升 4B（dense）。**

### 2.3 为什么不用 Llama/Gemma

`Llama 3.2 1B/3B`、`Gemma 3 1B/4B` 通用性强但中文偏弱，对中文电商槽位抽取不占优。

### 2.4 时间线说明（面试别讲错）

- `Qwen3`（0.6B/1.7B/4B）2025-04 开源，Apache 2.0；`Qwen3.5` 小模型（0.8B~9B）2026-03 开源，两者当前都可得；
- 都可得的情况下仍选 Qwen3 dense，理由是能力与负载对齐 + 部署生态成熟（TTFT、prefix caching、guided JSON、微调工具链全部踩熟）——讲「主动选型」比讲「当时没得选」更能体现选型功力。

---

## 3. 微调目标与指标（先量化、先跑基线）

| 指标 | 定义 | 说明 |
|---|---|---|
| 格式合规率 | 输出能否被 `JsonOutputParser` 解析成合法 TurnPlan | 越高越好 |
| 意图/flow_id 命中率 | `start_flow.flow` 是否正确 | eval 集标注 |
| 槽位 F1 | `set_slots.slots` 的 key/value 与人工标注匹配 | eval 集标注 |
| 澄清率 | validator 拒绝比例 | 越低越好 |

**铁律**：微调前必须用 `qwen-plus` 在同一个 eval 集上跑出基线。没有基线的微调，面试会被一句「你怎么证明是微调带来的提升」问死。

**防作弊红线**：澄清率下降的同时，槽位 F1 **不能**下降——防止模型靠「少输出槽位」来降低澄清率（变怂而不是变准）。

**validator 兜不住的才是真风险**：它只查 schema 层（命令类型、flow 存在性），查不出「flow 选错但存在」「槽位值幻觉」——这类错误 valid but wrong，澄清兜底不触发，小模型恰恰会放大。所以指标以人工标注的 eval 集为准，澄清率只作辅助观测。

---

## 4. 微调流程（前置 + 阶段 0~6）

### 前置阶段：项目先跑通 + 提示词落盘脚本（一鱼三吃，最关键的一步）

**微调本身只是一条命令，80% 的工作量在前面**。跑起来的 qwen-plus 版本有三个身份：体验基线、评测参照、训练数据的老师——它没跑起来，后面全是空中楼阁。

1. **项目跑通（1~2 天）**：建 MySQL 表 `dialogue_states`（`sender_id` 主键 + `state_json` TEXT）→ 写 `.env`（DashScope 申请 key）→ `uv run uvicorn atguigu.api.app:app --host 0.0.0.0 --port 18082` → Swagger 打 `/api/chat` 验证四条链路：业务消息走 task 轨道、闲聊走 chitchat、模糊话术触发澄清、`dialogue_states` 表有状态写入。电商中台不通不影响本阶段（TurnPlanner 链路不依赖它）；
2. **落盘脚本（3~5 天）**：在 `atguigu/test/` 下写离线 asyncio 脚本：手工构造各种 `DialogueState` → 复用 `TurnPlanner._build_prompt_inputs` + 渲染 `turn_plan.jinja2` → 完整提示词落盘 → 调 qwen-plus 拿输出落盘 → `TurnPlanValidator.validate` 质检。这一个脚本同时是：**评测集生成器**（人工标 50~100 条正确答案）、**训练数据流水线**（阶段 1 直接放大）、**换模型回归测试**（阶段 3 复用）；
3. **跑出基线**：qwen-plus 在 eval 集上跑出四指标（格式合规率 / 意图命中率 / 槽位 F1 / 澄清率）——这是要打败的数字。

### 阶段 0：环境

- 基座：`Qwen/Qwen3-1.7B`（ModelScope 同名仓库，国内直连快）
- 工具：ms-swift（ModelScope 出品，Qwen 官方生态）
- 硬件：**AutoDL 租单卡 24G（4090 首选 ≈2 元/时，3090 省钱档 ≈1.3 元/时，同容量都够用）**，跑 **bf16 LoRA**——租卡之后不为显存妥协，不用 QLoRA 的 4bit 量化噪声，合并权重更干净。1.7B 全量数据一轮（3 epoch）约 1~2 小时，**单次实验成本几元级**，可以放开手脚做对比实验
- 环境：AutoDL 选「PyTorch 2.x + CUDA 12.x」预装镜像，进去只装 ms-swift；模型走 ModelScope 自动下载，数据 jsonl（几 MB）用网盘/scp 上传；数据预处理、LoRA 合并这类轻活切**无卡模式**开机（约 0.1 元/时）更省；训完即关机，产物下载回本地
- 纪律不变：**先用 50 条样本跑通「数据→训练→合并→推理验证」全流程，再上全量**——按小时计费，更不能把时间浪费在数据格式错误上

```bash
# AutoDL 容器内
pip install "ms-swift[llm]" -U
```

### 阶段 1：数据准备（占 60% 工作量）

**样本格式（messages 格式，一条一对）**：

```json
{
  "messages": [
    {"role": "system", "content": "你是电商客服对话系统的意图路由规划器，只输出合法 JSON，不要输出 markdown 或解释。"},
    {"role": "user", "content": "<7 类上下文，用 turn_plan.jinja2 渲染后的完整内容>"},
    {"role": "assistant", "content": "{\"task\":{\"commands\":[{\"command\":\"start_flow\",\"flow\":\"logistics_tracking\"},{\"command\":\"set_slots\",\"slots\":{\"order_number\":\"A123456\"}}]},\"knowledge\":null,\"chitchat\":null}"}
  ]
}
```

**五条铁律**：

1. 训练时 `user` 内容 = 推理时 `turn_plan.jinja2` 渲染出的完整内容，保证训练/推理分布一致；
2. `assistant` 就是裸 JSON 字符串，schema 与 `TurnPlan.from_dict` 完全对齐；
3. **非思考模式渲染**：Qwen3-1.7B 是混合思考模型，样本必须以 `enable_thinking=False` 的 chat template 渲染，否则上线后会间歇性吐 `<think>` 思考链，`JsonOutputParser` 直接炸（这种错评测不刻意测发现不了）；
4. **只对 assistant 段算 loss**（completion-only）：input 2~3K token、output 几十 token，不 mask 掉 prompt 就是让模型花 95% 的梯度学背模板（ms-swift 默认只对 response 计损，勿关）；
5. 正反例都要有：validator 通过的是正例；被拒样本（`UNKNOWN_TASK_FLOW` / `MULTIPLE_TASK_FLOWS` / `INVALID_TASK_COMMANDS`）**不要原样入库当正确答案**——修到正确后入库，错误案例的价值是分析错误模式、针对性补数据。

**数据来源（闭环构造）**：

1. **离线状态工厂 + 模板渲染**：手搓各种 `DialogueState`，按覆盖矩阵拉齐——有无 `activated_task` / `paused_tasks` × 四种命令 × 单/多意图 × 纯闲聊 × 带卡片点击历史 × 不同历史长度（渲染 `turn_plan.jinja2` 得 input）。**缺一块覆盖，上线就崩一块**；
2. qwen-plus 蒸馏打标 → `TurnPlanValidator` 质检闸门 → 人工抽检（validator 查不出 valid-but-wrong，不能全信）；
3. 人工补边界样例：多意图、模糊意图、指代（"那个订单"）、情绪化表达、同义词槽位、resume/start 边界（"继续刚才那个" vs 新开业务）；
4. （上线后）真实日志回放回流，形成闭环。

**量级**：LoRA 起步 500 条先跑通流程，全量 3K~10K 条该任务即饱和；train 80% / eval 20%（**eval 集与训练集同分布但严格隔离，训练前建好、带人工标注**），存 `train.jsonl` / `eval.jsonl`。

### 阶段 2：训练（LoRA，bf16）

```bash
swift sft \
  --model Qwen/Qwen3-1.7B \
  --train_type lora \
  --dataset ./data/train.jsonl \
  --val_dataset ./data/eval.jsonl \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_target_modules ALL \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --max_length 4096 \
  --output_dir ./output/qwen3-1.7b-turnplanner
```

面试可讲：`r=16`/`alpha=32` 只训低秩增量、冻结基座 99%+ 参数，对路由类浅层映射任务容量足够（不给模型注入新知识，只重塑「模板→JSON」的条件分布边界）；序列长度 4096 覆盖模板 + 10 轮历史；4090 上一轮（3 epoch）约 1~2 小时；观察指标看 eval 格式合规率与意图/槽位指标，不只盯 loss。

### 阶段 3：评估（三组对比）

用同一个 eval 集对比：`qwen-plus`（基线） vs `Qwen3-1.7B 未微调` vs `Qwen3-1.7B 微调后`。

**通过标准**：澄清率 ↓ 且 槽位 F1 不降 且 意图命中率 ≥ 基线，否则回炉调数据/超参。若 1.7B 边际不达标：同配方把基座换 `Qwen3-4B` 再训一组（租卡成本几元级），1.7B/4B 同数据对比、以 eval 说话——算力可租之后，「先小后大」的升级路径几乎没有摩擦。

### 阶段 4：导出合并 + 部署

> ⚠️ **实战修正（2026-08-20）**：本阶段按下面写的 bf16 合并路线走**翻车了**——delta 量级 0.21% 小于 bf16 量化步长 0.4%，合并回存时 52.8% 的元素被舍回基座值，评测从 94%/71% 暴跌到 40%/0%（根因分析见 `turnplanner-bf16-merge-pitfall.md`）。
> **实际落地形态：跳过合并，vLLM `--enable-lora --lora-modules turnplanner=<checkpoint>` 运行时挂载 adapter**（浮点计算、与训练逐位一致），部署在 AutoDL，指南见 `turnplanner-deploy-autodl.md`。以下原文保留作为方案设计留档——若必须合并分发，改存 fp16 并做回放验收。

**A. 本地部署（契合「降本去依赖」叙事）**

两步走：合并 → 起服务。1.7B 合并后 bf16 权重仅 ~3.4GB，**笔记本 8G 卡直接跑得动**（剩余显存够 4K 上下文的 KV cache），默认无需量化；要压显存或在更小机器上跑，再可选 AWQ int4（~1.2GB）：

```bash
# 1. 合并 LoRA 回 bf16 全量权重
swift export \
  --model Qwen/Qwen3-1.7B \
  --adapters ./output/qwen3-1.7b-turnplanner/<checkpoint> \
  --merge_lora true --output_dir ./merged

# 2. vLLM 起本地 OpenAI 兼容服务
vllm serve ./merged --host 0.0.0.0 --port 18083

# 3.（可选）需要压显存时再量化 AWQ int4
swift export --model ./merged --quant_method awq --output_dir ./merged-awq
```

> 训练态的 adapter 别直接上线——先合并成独立权重再部署；量化是部署侧的可选项，不是训练的副产品。

**B. 云端托管（教学最省事）**：DashScope 上传数据集 → SFT → 一键部署，得到微调后 model id，直接当 `LLM_ROUTER_MODEL` 用。

### 阶段 5：接入项目（改动集中在三处）

**① `atguigu/config/config.py`** 加路由模型变量：

```python
llm_router_model: str      # 微调后的 Qwen3-1.7B（model id 或本地路径）
llm_router_base_url: str   # 本地 vLLM 端点，如 http://localhost:18083/v1
llm_router_api_key: str    # 本地服务填占位符
```

**② `atguigu/infrastructure/llm.py`** 把单例拆成两个 client：

```python
def _build_client(model: str, base_url: str, api_key: str):
    return init_chat_model(model=model, model_provider="openai",
                           base_url=base_url,
                           api_key=api_key,
                           temperature=0, timeout=60)

llm_client    = _build_client(settings.llm_model, settings.llm_base_url, settings.llm_api_key)                 # 生成/润色等，仍走 qwen-plus
router_client = _build_client(settings.llm_router_model, settings.llm_router_base_url, settings.llm_router_api_key)  # 微调后的路由小模型
```

**③ `atguigu/plan/planner.py`** 只改一行，把链里的 `llm_client` 换成 `router_client`：

```python
chain = prompt_template | router_client | JsonOutputParser()
```

其余 4 个调用点（clarify / knowledge / chitchat / action_response）**不动，继续用 qwen-plus**——这是「分层模型策略」。

> 注意：路由模型的 `base_url`/`api_key` **必须独立配置**（本地 vLLM 端点 + 占位 key），不能复用 `llm_base_url`——两个 client 指向不同服务。云端托管方案则填 DashScope 地址 + 微调后 model id。

### 阶段 6：灰切 + 监控 + 回滚

1. **灰度**：按 `sender_id` 抽样 10% 流量走微调模型，观察澄清率；
2. **监控**：看 `ClarifyReason` 分布，`UNKNOWN_TASK_FLOW` / `INVALID_TASK_COMMANDS` 是否下降；
3. **回滚**：异常时把 `LLM_ROUTER_MODEL` 改回 `qwen-plus` 即可，一行配置回滚，validator 和代码层零改动。

---

## 5. 面试话术（STAR）

### 5.0 电梯版（先背熟，数字为实测值）

> 我把系统里最高频、最该确定性的 LLM 调用点——路由规划器——从调大模型 API 改成了微调后的本地 Qwen3-1.7B 小模型，在保留三层防御不变的前提下，**格式合规率从 94% 提到 100%、语义一致率从 71% 提到 99%，双双反超大模型老师**；路由延迟从秒级降到几十毫秒、边际成本归零。中间还踩了个非常深的坑——LoRA 合并回 bf16 把微调信号量化抹掉了一半，靠教师强制 NLL 三方对比翻的案。

### 5.1 S — 背景与困境

> 这个电商客服系统的入口是 TurnPlanner：每条文本消息先过它，让 LLM 输出结构化 JSON（三轨道 + 命令 + 槽位），再由确定性 validator 校验。我当时看到两个问题：第一，路由是每条消息必走的最高频调用，输入 2~3K token、输出只有几十 token，延迟大头全耗在大模型 API 的 prefill 上，又贵又慢还有外部依赖；第二，尽管 prompt 已经写得非常严（裸 JSON、白名单、few-shot、temperature=0），澄清率还是偏高，主要集中在 flow_id 幻觉和命令格式错误上。

### 5.2 T — 目标

> 在不降低路由正确率、不破坏三层防御的前提下，把路由层从大模型换成可微调的小模型，同时把澄清率降下来。红线是 validator 这层绝对不能动——微调只动理解，不动裁决。

### 5.3 A — 动作（五步）

1. **选点**：评估全部 5 个 LLM 调用点，只有 TurnPlanner 满足「高频、结构化输出、有确定性兜底可当质检器」三条件，所以只微调它；
2. **数据闭环**：写离线状态工厂按覆盖矩阵构造 `(上下文 → TurnPlan)` 语料，qwen-plus 蒸馏打标，`TurnPlanValidator` 当质检闸门（通过才进候选池，被拒的人工分诊修复），再人工补多意图/指代/情绪化边界样例；
3. **评估先行**：建离线 eval 集，分意图/槽位/命令三层，先跑 qwen-plus 基线；
4. **训练**：AutoDL 租 24G 卡对 Qwen3-1.7B 做 bf16 LoRA 微调，两条纪律——非思考模式渲染防思考链污染 JSON、只对 completion 算 loss 防背模板；目标是「输出合法 JSON + 少触发 validator 拒绝」；
5. **接入灰切**：把单例 `llm_client` 拆成「路由小模型 + 生成大模型」双通道，OpenAI 兼容接口下只换 model id，灰度观察、异常一键回滚。

### 5.4 R — 结果（实测值）

> **格式合规率 94%→100%，语义一致率 71%→99%**，双双反超大模型老师；单轮路由成本趋近于零、延迟秒级→几十毫秒。最关键的两个附加收获：一是 eval 集上 flow_id 幻觉和命令格式错误全部消失（评测 100 条里 0 条格式错误）；二是排查合并事故沉淀了「合并产物必须做训练样本贪心回放验收」的流程铁律，validator 的 9 种原因码兜底逻辑一行没改。

（完整数字链：teacher 基线 94%/71% → 坏 merged 40%/0% → adapter 部署 100%/99%；NLL 对比 0.0004 vs 0.176；bf16/fp16 抹除率 52.8%/9.8%。细节见 `turnplanner-bf16-merge-pitfall.md` 与 `turnplanner-interview-story.md`。）

### 5.5 三个必被追问的「为什么」

**Q1：为什么微调，而不是把 prompt 写得更好？**
> prompt 已经接近极限（白名单、few-shot、temperature=0），剩下的是模型对小语种/模糊意图/格式约束的把握问题，只能内化到权重里。而且 prompt 越长 token 越贵；微调是把长 prompt 的规则蒸馏进小模型，推理时 prompt 反而能变短。

**Q2：怎么证明是微调的功劳？**
> 用同一个 eval 集、同一套 validator、同样的上下文，只换模型跑 A/B，唯一变量是权重。我看的是「澄清率降的同时槽位 F1 不能降」——防止模型靠少输出槽位作弊。

**Q3：微调会丢通用能力吗？为什么还留大模型？**
> 会，所以做多模型路由而不是全局替换。路由要确定、便宜、快，适合小模型微调；澄清润色、知识问答、闲聊这些开放式生成，数据分布无限、强依赖检索结果，微调性价比低，继续用大模型。这是分层模型策略。

### 5.6 选型类追问

**「为什么不用 9B / 为什么用 Qwen3-1.7B？」**
> 按任务复杂度 + 成本收益选型，不是越大越好。意图识别是结构化抽取任务，1B~3B 是甜点区，9B 多出的参数主要换通用生成能力，这个调用点用不上；且 9B 的显存/推理成本会推翻「高频、便宜、低延迟」的目标。原则是先用 1.7B 跑基线，用槽位 F1 和澄清率说话，数据证明不够再升 4B。

**「Qwen3.5 都出了，为什么还用 Qwen3 dense？」**
> 选型选的是「能力和负载对齐」，不是「更新的模型」。Qwen3.5 小模型的升级方向是 Hybrid 架构长文本、代码、多模态 Agent——TurnPlanner 输入几 K token、纯文本、输出几十 token JSON，跑分涨的维度一项都命中不了；而 Hybrid 架构的代价恰好全踩在关键路径上：TTFT 比 dense 慢（路由延迟大头就是 prefill）、Mamba 类缓存的 prefix caching 还是实验特性（我的 prompt 恰是「长稳定模板头 + 短多变尾」的 prefix cache 教科书场景）、4B 在 16GB 卡上部署 OOM。Qwen3.5 的 VL 能力留给后续「订单截图识别」需求，与路由层解耦。

**「小模型输出 JSON 不可靠怎么办？」**
> 双保险：训练侧 SFT 数据全是裸 JSON 的 completion（且非思考模式渲染）；部署侧 vLLM 起 OpenAI 兼容服务并开 guided JSON 约束解码，schema 合法性从「靠提示词求」变成「结构上不可能非法」。validator 仍是最后防线，三层防御不变。

**「都能租卡了，为什么不全参微调？」**
> 全参解决的是「注入新知识、大幅重塑行为」，我这个任务是浅层映射（模板→JSON），LoRA r=16 的容量就够。全参要求数据量更大（否则过拟合）、有灾难性遗忘风险（伤指令遵循和泛化）、单个 checkpoint 3.4GB 起步迭代笨重——LoRA adapter 只有几十 MB，实验敏捷。算力便宜了，更应该把预算花在「多组对比实验」上，而不是单组全参上。
