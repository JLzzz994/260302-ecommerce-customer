# TurnPlanner 微调翻车复盘：bf16 合并抹掉小量级 LoRA delta

> 2026-08-20 排查实录。现象：微调后模型上线路径全崩——eval 格式合规率 40%（teacher 基线 94%）、语义一致率 0%（基线 71%）、task 轨道场景全部输出三 null 或错误轨道、部分输出多一个 `}`。
>
> 结论先行：**训练完美、adapter 完美、败在 LoRA 合并回 bf16 权重这一步——delta 量级（|W| 的 0.21%）低于 bf16 的量化步长（相对 ~0.4%），52.8% 的元素在合并舍入中被直接抹回基座值。合并产物 ≈ 裸基座 + 半量化残渣。**

---

## 1. 排查链（每一步的排除法）

| # | 嫌疑 | 验证手段 | 结论 |
|---|---|---|---|
| 1 | WSL/vLLM 部署环境 | 本地 CPU transformers 前向复现同样错误输出 | ✗ 与环境无关 |
| 2 | 训练没收敛 | logging.jsonl：loss 1.48→0.0006，token_acc=1.0 | ✓ 训练正常 |
| 3 | 训练数据坏 | sft_train.jsonl 本地/服务器 sha256 一致，323 条全合法 JSON | ✗ 排除 |
| 4 | 下载断损 | merged-full 三个关键文件 sha256 双端一致 | ✗ 排除 |
| 5 | chat template 错位 | swift 源码：qwen3 模板 `non_thinking_prefix` 与部署模板逐字一致；train mode 编码 20 个 loss token 正常 | ✗ 排除 |
| 6 | 合并数学错 | 全模型 196 个 LoRA 张量 relerr≈0.0011，当时误判为"bf16 噪声" | 半对——这个"噪声"本身就是病灶 |
| 7 | **合并精度** | 见下节三组实验 | ✓✓✓ **根因** |

## 2. 三组定罪实验

**实验一：三方教师强制 NLL（swift 真实编码，同一 token 序列）**

| 模型 | t01 | t03 | t07 |
|---|---|---|---|
| base 裸基座 | 1.85 | 1.55 | 1.62 |
| **base + adapter-63（fp32 动态挂载）** | **0.0011** | **0.0004** | **0.0008** |
| merged-full（bf16 合并产物） | 0.196 | 0.176 | 0.236 |

adapter 版 NLL≈0.0004 与训练日志 0.0006 精确吻合——训练是真的；merged 版高了两个数量级——合并是坏的。

**实验二：贪心解码复现**

base + adapter-63（fp32）对 t01/t03/t07 三个训练样本**逐字复现训练目标**；merged-full 输出错误轨道（task 场景输出 knowledge/order_info）、格式带多余 `}`。

**实验三：bf16 舍入统计（q_proj 抽样层）**

```
delta 相对量级:          0.2143%
bf16 相对量化步长:       ~0.39%   (尾数 7 位)
bf16 合并后 delta 被完全抹掉的元素占比: 52.8%
fp16 同口径:             9.8%
bf16 合并保真 relerr:    0.001160  ← 就是实验六里被误判的"噪声"
```

`W + delta` 存回 bf16 时，delta 小于一个 ULP 的元素约一半被舍回原值。微调信号被量化逐元素摧毁，只留下方向性最强的部分（所以模型还记得"紧凑 JSON"格式——这是 delta 中幸存的最大方向，但语义分支点全部丢失）。

## 3. 为什么会踩这个坑

- LoRA r=16 / alpha=32 → scale=2，加上只有 323 条样本、63 步、loss 压到 0.0006——**训得越好越"轻"，delta 越小，越容易被 bf16 吃掉**。常规认知"合并 = 数学上 W+BA，无损"只对 fp32 成立；
- ms-swift export 默认按模型 dtype（bf16）保存合并权重，不会告警；
- 所有静态校验（张量对比、shape、哈希）都会"通过"——0.0011 的偏差看起来就是数值噪声，只有**行为级实验**（教师强制 NLL / 训练样本回放）能暴露。

## 4. 修复方案（按推荐排序）

1. **vLLM 原生 LoRA serving（推荐，零合并零上传）**：基座还在服务器上，adapter checkpoint-63 也还在，`--enable-lora` 动态挂载，运行时 fp32 计算，行为与训练精确一致。顺带绕开 merged 目录的 `extra_special_tokens` 坑（基座 tokenizer 本来就是好的）：

   ```bash
   vllm serve /root/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master \
     --host 0.0.0.0 --port 18083 --max-model-len 4096 \
     --enable-lora --max-lora-rank 16 \
     --lora-modules turnplanner=/root/autodl-tmp/output/full/v0-20260819-233903/checkpoint-63
   # 评测时 --model turnplanner
   ```

2. **fp16 合并**：抹除率 9.8%、保真 5 倍（fp16 尾数 10 位，ULP≈0.05% < delta 0.21%）。本地合并后上传 3.3G；
3. **重训时把 delta 做大**：`lora_alpha` 提到 64+（scale=4）或加大数据量/epoch，让 delta 量级显著超过 bf16 ULP 再合并——治本，但要重训。

## 5. 教训（面试可讲）

- "合并 LoRA 不是无损操作"——精度敏感任务里 bf16 合并的量化损失足以摧毁小量级微调，这是教科书不写、踩过才知道的坑；
- 诊断方法论：静态校验（哈希/张量/配置）全绿不代表没问题，**最终要以"训练样本能否被贪心解码复现"作为合并产物的验收标准**——这个测试在合并后 30 秒就能跑完，本可以第一天就发现问题；
- 三层防御（validator）挡不住这种错：输出仍是合法 JSON，validator 全放行——"valid but wrong"的教科书案例，eval 集人工标注才是最后的真话。

## 6. 后记：结局与归档（2026-08-20 晚）

- **修复落地**：弃 merged，vLLM `--enable-lora` 运行时挂载 checkpoint-63（AutoDL 部署，见 `turnplanner-deploy-autodl.md`）。最终评测 **100% 格式合规 / 99% 语义一致**，双双反超大模型老师（94%/71%）——数字链：`94/71 → 40/0 → 100/99`；
- **adapter 归档**：`D:\models\turnplanner-adapter-v1\`（权重 + 配置 + 档案 README），归档后本地挂载贪心回放 5 个训练样本全部逐字复现，副本完好可用；
- **流程沉淀**：合并产物上线前的「贪心回放验收」已写入部署指南与面试叙事；后续若需合并分发，fp16 + 回放验收双保险；
- **衍生面试金句**：训得越好、delta 越小，合并越危险（loss 压到 0.0006 恰好把自己推进 bf16 量化盲区）。
