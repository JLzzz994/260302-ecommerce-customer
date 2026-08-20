# TurnPlanner 微调项目会话总结（2026-08-19 ~ 08-20）

> 从「完全不会微调、项目刚写完没跑起来」到「微调模型上线反超大模型老师」的完整历程沉淀。
> 关联文档：`turnplanner-finetune.md`（方案）、`turnplan-dataset-log.md`（数据留档）、`turnplanner-bf16-merge-pitfall.md`（事故根因）、`turnplanner-deploy-autodl.md`（现行部署）、`turnplanner-interview-story.md`（面试叙事）、`turnplanner-deploy-wsl.md`（已废弃）。
>
> **最终成绩（08-20 晚）**：格式合规 100/100，语义一致 99/100 —— teacher 基线 94%/71%。中间经历 merged 权重 bf16 合并事故（40%/0%），根因与修复见下半场时间线。

---

## 一、项目背景与目标

- **项目**：基于 LLM 的电商客服对话系统（本项目 `atguigu`），三轨道架构（task/knowledge/chitchat），TurnPlanner 是每条文本消息必经的路由器，输出 TurnPlan JSON（轨道 + 命令 + 槽位）。
- **目标**：把 TurnPlanner 从 qwen-plus API 换成微调后的本地 Qwen3-1.7B，降成本/降延迟/降澄清率；红线：validator 确定性校验一行不改（微调只动理解，不动裁决）。

## 二、关键决策链（讨论定稿）

| 决策点 | 结论 | 核心理由 |
|---|---|---|
| 基座 | Qwen3-1.7B（备胎 Qwen3-4B dense） | 结构化 NLU 甜点区；第一论据是延迟/成本（prefill 占大头）；迁移成本的正确算法=蒸馏老师对齐+validator 零改动+澄清率分布要重跑基线 |
| 不用 Qwen3.5 | 能力与负载错位 | 升级方向（长文本/代码/VL/Agent）TurnPlanner 用不上；Hybrid 架构 TTFT 变慢、prefix caching 实验性、16G 卡 OOM；VL 能力留给订单截图需求 |
| 训练方法 | bf16 LoRA（AutoDL 租 4090） | 浅层映射任务 r=16 够；全参数据要求高+遗忘风险+迭代笨重；QLoRA 是 8G 卡妥协，租卡后不保留 |
| 数据策略 | 离线状态工厂 + qwen-plus 蒸馏 + validator 质检 + 人工核对 | validator 只能质检不能标注（查不出 valid-but-wrong）；eval 集必须独立人工核对 |
| 量级 | 训练 323 / 评测 100（人工已核对） | 覆盖矩阵 22 格 × 同义改写；t05/t09 翻车格加权；分布比总量重要 |
| 部署 | 训练在 AutoDL、推理在本地 WSL vLLM | 训练按小时计费用完关机；本地零成本常驻；OpenAI 兼容接口接入零代码改动 |

## 三、完成里程碑

1. **项目跑通**：.env（修了 base_url 多 `/chat/completions` 的 404）、MySQL、四条链路验证（task/chitchat/澄清/状态写入），含多轮状态推进验证；
2. **数据流水线**（`atguigu/test/build_turnplan_dataset.py`）：
   - 22 个基础场景（覆盖三轨道、四命令、恢复栈、过场、product 卡片、10 轮长历史、边界格）；
   - 同义改写 401 条（槽位值逐字校验 + 去重 + 加权），池 423 → **train 323 / eval 100**（分层切分）；
   - teacher 基线（修正 chitchat label bug 后）：**格式合规 94% / 语义一致 71%**；
   - 人工核对 100 条全过 + 引用合法性核查（intent/flow/槽位名全对）；
   - 修复：变体 chitchat label 的 `chat` 跟随变体文本（曾致 10 个百分点假错误）；
3. **训练**（AutoDL 4090）：
   - 干跑 50 条四环节全绿（训练 loss 1.713→0.114 / 合并 / vLLM 推理形状 / t04 语义验证）；
   - 全量 323 条 LoRA（r16/α32/lr1e-4/3epoch），合并导出 merged-full（3.4GB bf16）；
   - 实证：LoRA 未伤通用对话能力（压缩 prompt 下语义正确、格式自创——格式绑定与完整模板强关联，符合预期）；
4. **部署**（WSL vLLM）：黄金组合 `vllm 0.11.0 + transformers 4.57.1 + gcc(build-essential)` + tokenizer 补丁 + `--gpu-memory-utilization 0.85 --max-model-len 4096`，已走到 torch.compile 通过；
5. **评测脚本**（`atguigu/test/eval_turnplan_model.py`）：100 条并发打端点 → 解析（剥 think/围栏）→ schema 校验 → 四指标 + 分格子错误分布，与 teacher 基线同口径。

## 四、踩坑总账（12+ 坑，全记录在两份文档）

**数据/训练侧**：base_url 双拼 `/chat/completions` 404｜模型 id `Qwen3-1.7B-Instruct` 不存在（Qwen3 无 -Instruct 后缀）｜镜像 torch 2.3 < transformers 2.5 要求｜torch 2.13(cu13) 超驱动（570=CUDA12.8 上限）→ 钉 2.6.0｜torchvision 二进制不配套｜新版 swift 参数改名（`--tuner_type`、target_modules 集合语义）｜FSDP2 需 torch≥2.6｜训练机装 vllm 拆训练环境｜tensorboard×numpy2.0（只影响收尾画图）｜改写 API 超时打断 gather → 逐批重试。

**部署侧**：vllm 0.27 V2 runner 在 WSL 报 UVA → 0.11.0｜transformers v5 缺属性 / 4.56 被 Qwen3 新 tokenizer 配置炸 → 4.57.1｜`extra_special_tokens` list → 置 `{}`（Python 改 JSON，勿用 sed）｜裸 WSL 无 gcc → build-essential（apt 换阿里云源提速）｜8G 显存：Windows 侧占用 ≤1.5G + `0.85` 参数｜uv venv 无 pip → `uv pip install --python` + 绝对路径｜venv/模型放 /mnt/* 慢 5~10 倍 → 全放 `~/`｜PyPI 超时 → 清华源。

**评测侧**：eval 抢跑（服务没到 `Application startup complete`）→ 10061 拒绝连接，0% 是链路错误不是模型错误；mirrored 模式已确认，剩余排查点=服务就绪时机/防火墙放行 18083。

## 五、下半场时间线（08-20 下午～晚间）：40%/0% 事故到 100%/99% 闭环

上午的卡点（WSL 服务起不来）解决后，冒烟评测跑出**灾难性数字：格式合规 40% / 语义一致 0%**（teacher 基线 94%/71%）。由此展开的全链路排查与翻案：

1. **排除环境**：本地 CPU + transformers 前向复现同样错误输出（t03 输出 knowledge/order_info、多余 `}`），证明与 WSL/vLLM 无关；
2. **排除训练**：训练日志 loss 1.48→0.0006、token_acc 1.0 完美收敛；
3. **排除数据/传输**：sft_train.jsonl 与 merged-full 关键文件双端 sha256 逐字节一致；
4. **排除模板**：读 ms-swift 4.5.2 源码，qwen3 模板 non_thinking_prefix 与部署模板逐字一致；train mode 编码 loss 掩码正常；
5. **走了一段弯路**：张量级重建对比 196 个 LoRA 层误差 0.0011，误判为"bf16 舍入噪声"放行——这个数恰是损伤本身；
6. **行为实验翻案**：swift 真实编码 + 三方教师强制 NLL——裸基座 1.55 / **base+adapter 0.0004（与训练日志吻合）** / **merged 0.176**；adapter 贪心解码逐字复现训练目标。训练是好的，合并是坏的；
7. **根因**：LoRA delta 量级 0.21% < bf16 相对量化步长 0.4%，回存时 **52.8% 元素被舍回基座值**；fp16 同口径仅 9.8%。反直觉推论：训得越好 delta 越小，合并越危险。详见 `turnplanner-bf16-merge-pitfall.md`；
8. **修复（不重训）**：弃 merged，vLLM `--enable-lora` 运行时挂载 checkpoint-63 adapter（浮点计算，与训练逐位一致）。

随后完成部署与接入（期间 WSL 环境整体清除废弃）：

- **AutoDL 服务器侧装环境**：实例对外限速 ~0.3MB/s，改走「Windows 下载 wheel → SFTP 上传 → 离线安装」；后又发现 Windows 下载机会静默跳过 Linux 平台条件依赖（nvidia/cudnn/triton/xformers 等 18 个），最终改由服务器侧 pip 直下补齐（aliyun 源通畅）。坑位：pip 装 CPU 版 torch 抢位需 force-reinstall CUDA 版；nvidia 库不随 CPU torch 安装需手动装齐；
- **部署**：vLLM serve 基座 + `--enable-lora --lora-modules turnplanner=checkpoint-63`，`--api-key` 鉴权，AutoDL 公网 HTTPS 映射对外（详见 `turnplanner-deploy-autodl.md`）；
- **项目接入**（分层模型策略落地）：config.py 加 router 三变量、llm.py 拆 `llm_client`/`router_client` 双通道（后者固定 `enable_thinking=False` 与训练渲染对齐）、planner.py 换 router_client；其余四个 LLM 调用点不动。回滚 = `.env` 三行改回大模型；
- **评测**：`eval_turnplan_model.py` 补 Bearer 鉴权 → 冒烟 5/5 → **全量 100/100 格式 + 99/100 语义**。唯一 miss：t21 多意图变体漏抽第二个 intent（数据覆盖问题，非模型缺陷，下轮补变体即可）；
- **归档**：adapter 存 `D:\models\turnplanner-adapter-v1\`（含 README 档案），归档后本地挂载贪心回放 5/5 复现验证。期间 AutoDL SSH 服务挂过一次（公网推理映射不受影响），靠下午下载的本地副本完成归档。

## 六、当前状态与下一步

**状态：本项目微调线全部闭环。** 路由走微调模型（99% 语义一致反超基线），表达层走大模型，回滚一行配置。

待办（低优先）：
1. t21 多意图补 3~5 条变体，出 v2 adapter（流水线现成，几元成本）；
2. AutoDL 实例管理：SSH 已挂，需控制台重启（公网地址会变，同步改 `.env`）；长期挂业务考虑把 vLLM 挪回自有机器（归档目录直接 `--lora-modules` 复活）；
3. 若未来需要 merged 分发：fp16 合并 + 回放验收（铁律见 pitfall 文档）。

## 七、可直接复用的资产

| 资产 | 位置 | 一句话 |
|---|---|---|
| 数据集流水线 | `atguigu/test/build_turnplan_dataset.py` | 场景工厂+改写+质检+切分，重跑一轮几元 |
| 评测脚本 | `atguigu/test/eval_turnplan_model.py` | 任何模型的四指标+分格子（已带 Bearer 鉴权，默认打公网映射） |
| 训练数据 | `data/turnplan/sft_train.jsonl`（323） | ms-swift messages 格式 |
| 评测集 | `data/turnplan/eval_set.jsonl`（100） | 已人工核对 |
| teacher 基线 | `data/turnplan/teacher_eval.jsonl` | 94% / 71%，同 id 可 diff |
| 微调模型评测 | `data/turnplan/model_eval.jsonl` | **100% / 99%**，同 id 可 diff |
| **模型归档（真身）** | `D:\models\turnplanner-adapter-v1\` | checkpoint-63 adapter + 档案 README，回放 5/5 验证 |
| 基座模型 | `D:\models\Qwen3-1.7B-base\`（另有 AutoDL modelscope 缓存） | 复活归档所需 |
| ~~merged-full~~ | D 盘 / 数据盘 | **已废弃**：bf16 合并损伤，勿部署 |
| 文档 | `docs/` | 方案 / 数据留档 / 事故根因 / 现行部署 / 面试叙事 / 废弃部署 |

## 八、面试可讲的叙事线（STAR 浓缩）

**S**：路由是最高频调用，大模型 API 贵/慢/外部依赖；**T**：不降正确率、三层防御不动，换成本地小模型；**A**：五步——选点（5 个调用点只选满足三条件的 TurnPlanner）→ 数据闭环（状态工厂+蒸馏+validator 质检+人工核对，实证 teacher 偏差 29% 且含 valid-but-wrong）→ 评估先行（100 条 eval + 94%/71% 基线）→ LoRA 训练（租卡后不为显存妥协）→ **事故排查（bf16 合并抹除 52.8% delta，行为实验翻案）** → 双通道接入可回滚；**R**：**格式合规 94%→100%、语义一致 71%→99%，双双反超大模型老师；路由延迟秒级→几十毫秒，边际成本归零**。

加分素材：teacher 在 resume/set_slots 边界、多槽位抽取上翻车的实证；validator 兜不住 valid-but-wrong 的活例（bf16 事故输出合法 JSON 全过校验）；chitchat label bug 致基线虚低 10 个百分点被人工核对发现——「评测基建可信度」的最佳论据。完整话术见 `turnplanner-interview-story.md`。
