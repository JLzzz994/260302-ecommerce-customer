# TurnPlanner 数据集构建留档

> 更新：2026-08-19 第二轮（扩格子 + 扩量 + eval 基线）｜ 脚本：`atguigu/test/build_turnplan_dataset.py` ｜ 产物：`data/turnplan/`

## 第二轮：扩格子 → 扩量 → eval 基线（2026-08-19）

### 执行记录

| # | 做了什么 | 命令 | 结果 |
|---|---|---|---|
| 1 | 新增 8 个未覆盖格子（t15~t22），`_build_state` 支持 `system_task`（过场上下文） | 无参渲染验证 | 22 基础场景全部渲染通过，轨道分布 task 14 / knowledge 4 / chitchat 2 / 澄清 1 / 多意图 1 |
| 2 | 同义改写扩量（每场景 20×权重，LLM 生成 + 槽位值逐字校验 + 去重） | `--expand 20 --eval-count 100 --dryrun 50` | 变体 401 条，池 423 = 22 基础 + 401 变体；**训练 323 / 评测 100**（按父场景分层轮转切分）；干跑包 50 条 |
| 3 | eval 集 teacher 基线（100 条并发 + validator 质检 + label 比对） | `--reuse-variants --teacher-eval` | **格式合规 91% / 语义一致 61% / 解析失败 5%** |

### 第一轮遇到的问题与修复

- 改写批次 API 超时（60s timeout）会打断整个 gather → 已改为**逐批重试 3 次 + 失败只告警不中断**，最终 t09/t22 各有一个批次仍失败被丢弃（可接受，量已够）；
- 后台跑命令用管道 `| tail` 会返回假 exit 0 → 改为重定向日志文件再检查。

### 人工核对结论（2026-08-19，用户完成）

100 条 eval label 全部人工核对通过；引用合法性核查通过（4 个 intent id 均在 `KNOWLEDGE_INTENTS` 且 requires_object 与卡片匹配；5 个 flow_id 与 order_number/refund_reason 槽位名在 user_flows.yml 全部存在）。

核对中发现一个真 bug：**变体的 chitchat label 里 `chat` 沿用了父场景原话**（下游 `ChitChatResponder` 拿 chat 当当前用户消息回传，必须跟随变体文本）。已修复：`expand_variants`/`load_variants` 统一走 `_variant_label()`（深拷贝 + 替换 chat），`--reuse-variants` 复现重导出（未重新调 LLM 改写，零成本）。

### 修正后的 teacher 基线（以这组为最终基线）

| 指标 | 修正前 | 修正后 | 变化 |
|---|---|---|---|
| 格式合规率 valid | 91% | **94%** | +3 |
| 语义一致 match | 61% | **71%** | +10（全为 chitchat 假错误） |
| 解析失败 | 5 条 | **1 条** | -4 |

不一致 29 条按父场景分布：t08 带卡片知识/task 分流 5/5、t13 模糊话术 4/4、t15 相似推荐 4/4、t18 过场中补槽位 4/4、t20 打断开新流程 3/3、t22 栈空恢复 3/5、t12 内置恢复 2/3、t09 收集中补槽位 1/5、基础场景 3/9。**这些就是微调后要重点对比的格子。**

### 当前产物

| 文件 | 行数 | 用途 |
|---|---|---|
| `inputs.jsonl` | 22 | 基础场景考卷（评测/回归输入） |
| `variants.jsonl` | 401 | 变体话术（可 `--reuse-variants` 复现，不重复花 LLM 钱） |
| `sft_train.jsonl` | 323 | **训练数据**（ms-swift messages 格式） |
| `eval_set.jsonl` | 100 | **评测集**（带 label，待人工核对） |
| `teacher_eval.jsonl` | 100 | 老师基线逐条结果 |
| `dryrun_50.jsonl` | 50 | AutoDL 干跑包 |
| `expand_run.log` / `teacher_eval_run.log` | - | 运行日志 |

### AutoDL 干跑指南（dryrun_50.jsonl）

**环境踩坑实录（2026-08-19，AutoDL 4090 + PyTorch2.3/CUDA12.1 镜像，按序执行可一次通过）**：

```bash
pip install "ms-swift[llm]" -U -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
pip install "torch==2.6.0" "torchvision==0.21.0" -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
pip install -U tensorboard   # 修 numpy2.0 兼容（收尾画图用，不修只影响日志美观）
```

踩坑对照表（现象→根因→修法）：

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | transformers 报 `PyTorch >= 2.5 is required but found 2.3.0` | 镜像自带 torch 太旧 | 升 torch |
| 2 | 升完 torch 报 CUDA 驱动不足 | 默认源拉到 CUDA13 构建，驱动 570 只到 12.8 | 钉 `torch==2.6.0`（PyPI 默认 cu124 构建） |
| 3 | `operator torchvision::nms does not exist` | torchvision 0.18 按旧 torch 编译 | 配套升 `torchvision==0.21.0` |
| 4 | `cannot import name 'FSDPModule'` | 新 ms-swift 要 torch≥2.6（FSDP2 API） | 同 2，一步到位 2.6.0 |
| 5 | ModelScope 404 | `Qwen3-1.7B-Instruct` 不存在（Qwen3 无 -Instruct 后缀） | 改用 `Qwen/Qwen3-1.7B` |
| 6 | `Target modules {'all'} not found` / remaining_argv | 新版 swift 不再翻译 ALL 等别名；`--train_type` 改名 | 删掉 target_modules 用默认值（默认即七模块全挂）；`--tuner_type lora` |
| 7 | 训练实例上装 vllm 后 torch 被顶到 2.13（cu13） | vllm 0.27 依赖新 torch，驱动 570 跑不了且破坏训练环境 | **训练机不装 vllm**（部署在本地 WSL）；若已装需重装 torch==2.6.0 + torchvision==0.21.0 才能再训练 |
| 8 | WSL 上 vllm 0.27 报 `UVA is not available`（V2 Model Runner 初始化） | vllm 0.27 的 V2 runner 与 WSL CUDA 驱动不兼容，与显存/参数无关 | 降级 `pip install vllm==0.11.0`（V1 引擎稳定版）；另：8G 笔记本需 `--gpu-memory-utilization 0.85 --max-model-len 4096`，Windows 侧 GPU 占用需 ≤1.5G |
| 9 | vllm 0.11 启动报 `Qwen2Tokenizer has no attribute all_special_tokens_extended` | venv 里残留 transformers v5（装 0.27 时带入），与 0.11 不配套 | `uv pip install transformers==4.56.0 --python <venv>/bin/python`；**WSL 部署最终版本组合：vllm 0.11.0 + transformers 4.56.0 + torch 2.8** |
| 10 | WSL 里 `(turnplan-env)` 激活了但 `which pip` 指向 `/usr/bin/pip` | uv 创建的 venv 默认不含 pip，activate 后 PATH 回落到系统 pip | 统一用 `uv pip install --python <venv>/bin/python` 装包、用绝对路径 `<venv>/bin/vllm` 起服务，不依赖 activate |

**验证过的训练命令（可直接复用）**：

```bash
swift sft \
  --model Qwen/Qwen3-1.7B \
  --tuner_type lora \
  --dataset ./dryrun_50.jsonl \
  --lora_rank 16 --lora_alpha 32 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 --lr_scheduler_type cosine --warmup_ratio 0.05 \
  --max_length 4096 \
  --output_dir ./output/dryrun
```

**干跑训练结果（2026-08-19 通过）**：12/12 步 37 秒，loss 1.713→0.114，token_acc 0.99，显存 4.73G，LoRA 挂载默认七模块（q/k/v/o/gate/up/down_proj），可训练参数 17.4M（1.003%）。日志确认 `[-100 * 1393]` 前缀——提示词部分 loss 被 mask，只对 JSON completion 训练，符合铁律。

租卡后完整流程（镜像选 PyTorch 2.x + CUDA 12.x，单卡 4090/3090 均可）：

```bash
# 1. 装 ms-swift
pip install "ms-swift[llm]" -U

# 2. 上传 data/turnplan/dryrun_50.jsonl（AutoDL 网盘 / scp / jupyter 上传均可）

# 3. 干跑训练（50 条 × 3 epoch，几分钟级，目的：验证数据格式与链路，不看效果）
swift sft \
  --model Qwen/Qwen3-1.7B \
  --train_type lora \
  --dataset ./dryrun_50.jsonl \
  --lora_rank 16 --lora_alpha 32 --lora_target_modules ALL \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 --lr_scheduler_type cosine --warmup_ratio 0.05 \
  --max_length 4096 \
  --output_dir ./output/dryrun

# 4. 合并 + 快速验证（合并后在 AutoDL 上直接 inference 试几条）
swift export --model Qwen/Qwen3-1.7B \
  --adapters ./output/dryrun/<checkpoint> --merge_lora true --output_dir ./merged-dryrun

swift infer --model ./merged-dryrun \
  --infer_backend pt \
  --adapters ./output/dryrun/<checkpoint>

# 5. 干跑产物下载回本地，验证通过后再上全量 323 条（同样命令换 dataset 路径）
```

通过标准：loss 正常下降、无 OOM、infer 输出是合法 TurnPlan JSON。注意思考模式：
Qwen3-1.7B 推理时若吐 `<think>`，正确做法是数据侧保证非思考渲染（本数据集 assistant 为纯 JSON），
若 infer 出现思考链再排查 swift 的 chat template 参数（如 `--loss_scale`/template 相关配置）。

### 待办

1. AutoDL 干跑：`dryrun_50.jsonl` 按指南跑通全流程；
2. 干跑通过后全量 323 条正式训练，eval 集测四指标对比 teacher 最终基线（71%）；
3. 训练样本目前 user+assistant 无 system 消息（对齐 planner.py 真实链路），待同步 `docs/turnplanner-finetune.md` 阶段 1 示例。

---

## 第一轮：14 场景起步（2026-08-19）

## 一、执行步骤记录

| # | 做了什么 | 命令 | 结果 |
|---|---|---|---|
| 1 | 渲染落盘 14 个起步场景 | `uv run python -m atguigu.test.build_turnplan_dataset` | 14 条写入 `inputs.jsonl`，轨道分布：task 8 / chitchat 2 / knowledge 2 / 澄清 1 / 多意图 1 |
| 2 | 抽查渲染质量（t11 恢复场景） | 内嵌校验代码 | `Active Task`=logistics_tracking、`Interrupted Tasks`=[refund_request]、两轮历史完整，提示词 ~3.8K 字符 ✅ |
| 3 | 全量 teacher 打标 + validator 质检 | `--teacher` | 14 条完成，**9 条 OK / 5 条 DIFF**（详见下节） |
| 4 | 逐条比对 DIFF 原始 JSON | 内嵌校验代码 | 5 条全部定性：label 均站得住，DIFF 均为老师模型偏差 |
| 5 | 导出 SFT 训练格式 | `--export-sft` | 14 条写入 `sft_train.jsonl`，assistant 为紧凑裸 JSON，user 为渲染后完整模板（~3.7K 字符） |

## 二、5 条 DIFF 逐条定性（label 全部保留，不改）

| 场景 | 老师输出 vs label | 定性 |
|---|---|---|
| t05 退款多槽位 | 老师漏抽 `refund_reason:"太贵了"` | **老师错**（valid-but-wrong 活例：schema 合法但语义漏抽，validator 查不出来） |
| t08 带卡片订单咨询 | 老师开物流流程 vs label 走 `order_info` 知识 | **label 符合设计**（focused_object 上下文 + 询问状态 = 知识咨询；"还没送到"有物流色彩，属边界 case） |
| t09 收集中补槽位 | 老师多输出 `resume_flow` | **老师错**（activated_task 在 collect 中只需 `set_slots`；resume 是恢复 paused 栈的——典型 resume/set_slots 边界混淆，连老师模型都犯，训练数据必须重点覆盖此格子） |
| t12 内置恢复 | 老师输出 `resume_flow`+flow（精确） vs label 不带 flow（内置） | **语义等价**（只有一个 paused task，两种写法系统行为一致）；按提示词规范"未指定则省略"，label 更规范 |
| t13 模糊话术 | 老师路由 chitchat vs label 全 null（触发澄清） | **产品决策**：训练目标是模糊话术触发正式澄清链路而非闲聊兜住；两者都讲得通，以 label 为准 |

另外 t14（多意图）`valid=False multiple_tracks` 是**预期行为**：模型如实输出双轨道 → validator 拒绝 → 澄清。这正是三层防御的设计演示，不是错误。

**本轮最重要结论**：teacher 5/14 偏差率（36%），其中至少 2 条是 validator 兜不住的语义错误——实证了"蒸馏数据不能全信老师，人工 label 是金标准"，也实证了建立带人工标注 eval 集的必要性。

## 三、产物文件清单

| 文件 | 内容 | 用途 |
|---|---|---|
| `inputs.jsonl` | 每行 `{id, name, user_message, prompt, label}`，prompt 为渲染后完整提示词 | 评测/回归测试的统一输入 |
| `teacher.jsonl` | 每行 `{id, teacher_output, valid, reason, match_label, error}` | 老师模型（当前 DeepSeek-V4-Flash）基线，即未来"三组对比"的基线数据来源 |
| `sft_train.jsonl` | ms-swift messages 格式（user+assistant），metadata 携带场景 id | 训练数据（当前 14 条起步，待扩量） |

## 四、关键决策记录

1. **训练样本不加 system 消息**：`planner.py` 的链路是 `PromptTemplate | llm_client | JsonOutputParser`，推理时就是单条 user 消息（渲染后模板）。训练样本保持 user+assistant 同构，分布才一致。（注：`docs/turnplanner-finetune.md` 阶段 1 的示例带 system，以本决策为准，待同步文档。）
2. **assistant 为紧凑裸 JSON**（`separators=(",",":")`），schema 与 `TurnPlan.from_dict` 对齐，null 字段保留。
3. **label 比对做归一化**：值为 None 的 key 视为缺省等价（`resume_flow` 的 `flow=None` 与不写等价）。

## 五、下一步

（已于第二轮完成，见文首「第二轮」章节；后续计划以第二轮的「待办」为准。）
