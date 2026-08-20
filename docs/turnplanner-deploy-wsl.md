# TurnPlanner 推理服务部署指南（WSL + vLLM）

> ⛔ **本文档已废弃（2026-08-20）**——WSL 环境（venv/模型/缓存）已整体清除，且文中部署的 `merged-full` 权重后被证实是坏的（bf16 合并抹除 52.8% 的 LoRA delta，见 `turnplanner-bf16-merge-pitfall.md`）。
>
> **现行部署**：AutoDL vLLM `--enable-lora` 运行时挂载 adapter，见 `turnplanner-deploy-autodl.md`。
> 本文档价值：① 12 坑速查表仍适用于任何 WSL vLLM 场景；② 作为「8G 显存挤压部署」的踩坑历史留档。版本组合 `vllm 0.11.0 + transformers 4.57.1` 的结论在新指南中继续沿用。

> 目标：在笔记本 Windows 的 WSL 里，用 vLLM 部署微调后的 Qwen3-1.7B（merged-full），为 TurnPlanner 提供 OpenAI 兼容的本地推理端点。
>
> 本指南由 2026-08-19/20 实际部署踩坑提炼，**每一步的参数都是验证过的**，按序执行可一次通过。

---

## 0. 架构与分工

| 组件 | 位置 | 说明 |
|---|---|---|
| 训练 | AutoDL 租卡（4090） | ms-swift LoRA，用完关机 |
| 模型产物 | `merged-full/`（~3.4GB bf16） | AutoDL 下载 → Windows D 盘 → 拷入 WSL |
| 推理服务 | 本机 WSL（3070 8G） | vLLM，端口 18083 |
| 业务服务 | Windows 本体 | FastAPI，`.env` 指向 `http://localhost:18083/v1` |

WSL 重置后：WSL 内的一切（旧 venv、~/models）已清空，需从第 1 步重来；Windows 侧的 D 盘模型和项目代码不受影响。

---

## 1. 前置检查（Windows 侧 + WSL 侧）

**Windows 侧**：
- NVIDIA 驱动已装（设备管理器/官网确认），**WSL 里不需要也不允许再装 Linux 驱动**，不需要装 CUDA Toolkit（vLLM 的 pip 包自带 CUDA 运行时）；
- `merged-full` 模型目录在 D 盘某处，包含 `config.json` + `*.safetensors`（合计约 3.4GB）+ `tokenizer.json` + `tokenizer_config.json` + `generation_config.json`。

**WSL 侧**（进入 Ubuntu 终端）：

```bash
nvidia-smi    # 能看到 RTX 3070 和驱动版本即正常，驱动 ≥ 525
```

看不到 GPU 就回 Windows `wsl --update` 后重开终端。

---

## 2. 装 uv + 建 venv（关键：放 WSL 原生目录，不要放 /mnt/c）

> 坑：venv 建在 /mnt/c（Windows 盘）上，import/启动慢 5~10 倍（9p 协议跨文件系统 IO）。

**先装系统依赖（裸 WSL 缺 C 编译器，triton 编译内核必需）**：

```bash
sudo apt update && sudo apt install -y build-essential
```

```bash
# 装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc    # 或重开终端让 PATH 生效
uv --version        # 确认可用

# 建 venv（Python 3.11，原生目录）
uv venv ~/turnplan-env --python 3.11
```

> 坑：uv 创建的 venv **默认不含 pip**，activate 后 `which pip` 可能落到系统 pip（还带 externally-managed 保护）。**统一用 `uv pip install --python <venv>/bin/python` 装包、用绝对路径起服务，不依赖 activate。**

---

## 3. 安装 vLLM + transformers（版本是关键）

> 坑1：vllm 最新版（0.27.x）的 V2 Model Runner 在 WSL 上报 `UVA is not available`，与显存参数无关，**必须用 0.11.0**（V1 引擎稳定版）。
> 坑2：transformers v5 缺 `all_special_tokens_extended` 属性（vllm 0.11 用），但 4.56.0 又会被 Qwen3 新版 tokenizer 配置的 `extra_special_tokens`(list) 炸出 `'list' object has no attribute 'keys'`——**钉 4.57.1**（两头兼容）。
> 坑3：PyPI 直连超时，**必须带清华源**。

```bash
uv pip install "vllm==0.11.0" "transformers==4.57.1" \
  --python ~/turnplan-env/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

一劳永逸（可选，以后装包不用再带源参数）：

```bash
mkdir -p ~/.config/uv
echo '[[index]] url = "https://pypi.tuna.tsinghua.edu.cn/simple" default = true' >> ~/.config/uv/uv.toml
```

**验证版本组合**（WSL 部署的黄金组合：`vllm 0.11.0 + transformers 4.57.1 + torch 2.8`）：

```bash
~/turnplan-env/bin/python -c "import vllm, transformers; print(vllm.__version__, transformers.__version__)"
# 期望：0.11.0 4.57.1
```

---

## 4. 拷贝模型（从 D 盘到 WSL 原生目录）

> 坑：模型放 /mnt/d 上直接 serve 会拖慢权重加载，**必须拷进 WSL 文件系统**。

```bash
mkdir -p ~/models
cp -r /mnt/d/<你的路径>/merged-full ~/models/

# 完整性校验（重要：下载/拷贝断损会到 serve 时才炸）
ls -lh ~/models/merged-full/    # 应有 config.json / *.safetensors / tokenizer* 等
du -sh ~/models/merged-full/    # 总量约 3.4G
```

---

## 5. 释放显存（8G 卡的生死线）

> 坑：8G 显存由 Windows 桌面和 WSL 共享。Windows 侧占用 3G+ 时，vLLM 初始化会以各种面目失败（显存不足报 `GPU memory utilization` 错，紧张时报 UVA 错）。

操作：Windows 任务管理器 → 进程 → 右键列头 → 勾选**「专用 GPU 内存」**列（默认的 GPU 列是 3D 占用率，看不出显存！）→ 排序找大户 → 关浏览器（或禁硬件加速）、微信、游戏等，把**专用 GPU 占用压到 1.5GB 以下**。

WSL 里验证：

```bash
nvidia-smi    # memory used 应在 1.5G 以内（含 Windows 侧占用）
```

---

## 6. 启动服务（验证过的完整命令）

```bash
~/turnplan-env/bin/vllm serve ~/models/merged-full \
  --host 0.0.0.0 --port 18083 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096
```

参数说明：
- `--gpu-memory-utilization 0.85`：8G 卡默认 0.92 会超（Windows 已占一块），0.85 = 上限 6.8G，权重 3.4G + KV cache 余量充足；
- `--max-model-len 4096`：模型默认按 40960 开 KV 预算纯属浪费，TurnPlanner 提示词最长 ~1600 token，4096 足够；
- 若显存实在紧张（0.85 报不足），低配档：`--gpu-memory-utilization 0.70 --enforce-eager`（跳过 CUDA graph，省几百 MB，推理略慢，对本场景无感）。

**启动预期**（原生目录 + 已编译过缓存后约 1~2 分钟；首次 3~5 分钟）：

```
Resolved architecture: Qwen3ForCausalLM
Using max model len 4096
... loading weights ...
... Capturing CUDA graphs ...
INFO ... Uvicorn running on http://0.0.0.0:18083
INFO ... Application startup complete      ← 看到这行才算就绪
```

served model name = 模型完整路径 `/home/<用户名>/models/merged-full`（调 API 时 model 字段用它，或加 `--served-model-name turnplanner` 自定义短名）。

---

## 7. 验证与评测（Windows 侧执行）

**冒烟 5 条**：

```bash
cd /d/大模型0301/teacher_code/260302-ecommerce-customer
uv run python -m atguigu.test.eval_turnplan_model --limit 5 \
  --model /home/administrator/models/merged-full
```

> 首批请求偏慢是正常预热（懒加载 + shape 特化）。若报 model not found，`--model` 换成 vLLM 启动日志里显示的名字，或在启动时加 `--served-model-name turnplanner` 后用 `--model turnplanner`。

**全量评测（100 条，对比 teacher 基线：格式合规 94% / 语义一致 71%）**：

```bash
uv run python -m atguigu.test.eval_turnplan_model \
  --model /home/administrator/models/merged-full
```

产物：`data/turnplan/model_eval.jsonl`（与 `teacher_eval.jsonl` 同 id 对齐，可逐条 diff）。

---

## 8. 日常使用

**起服务**（建议加 alias 到 `~/.bashrc`）：

```bash
echo "alias vllmserve='~/turnplan-env/bin/vllm serve ~/models/merged-full --host 0.0.0.0 --port 18083 --gpu-memory-utilization 0.85 --max-model-len 4096'" >> ~/.bashrc
source ~/.bashrc
# 以后一条命令：vllmserve
```

**接入业务**（评测达标后）：按 `docs/turnplanner-finetune.md` 阶段 5——`config.py` 加 router 三变量、`llm.py` 拆双 client、`planner.py` 换 `router_client`，`.env` 路由指向 `http://localhost:18083/v1`。**不要直接改 `LLM_BASE_URL`**（全局单 client 会把闲聊/澄清/知识也打到 1.7B 上）。

**换新模型**：AutoDL 训完 → 合并 → 下载 D 盘 → `rm -rf ~/models/merged-full && cp -r /mnt/d/.../merged-full ~/models/` → 重启服务（alias 同一条）。

---

## 9. 备选：AutoDL 云端部署（快车道）

WSL 折腾受阻时的替代方案（或需要快速评测时）：

```bash
# AutoDL 实例（PyTorch 2.x + CUDA 12.x 镜像；旧实例里还有 merged-full 就直接用）
pip install "vllm==0.11.0" "transformers==4.57.1" \
  -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com

# tokenizer 修复同样要打（extra_special_tokens → {}，见坑12）

vllm serve /root/autodl-tmp/merged-full --host 0.0.0.0 --port 18083 --max-model-len 4096
# 4090 24G 无需压显存参数；控制台开端口映射(18083)拿外网地址
```

注意：**最新 vLLM（0.27/torch2.13-cu13）在 AutoDL 驱动 570 上起不来**（需驱动≥580），别追新，用验证过的组合。成本约 2 元/时：适合评测与联调（十来块），不适合长期 7×24（日均 40+ 元——长期部署回 WSL/本地）。

---

## 10. 坑清单速查表

| # | 症状 | 根因 | 解法 |
|---|---|---|---|
| 1 | WSL 里 nvidia-smi 找不到 GPU | 驱动/WSL 版本旧 | Windows `wsl --update` + 装新驱动 |
| 2 | `UVA is not available` | vllm 0.27 V2 runner 与 WSL 不兼容 | **vllm==0.11.0** |
| 3 | `all_special_tokens_extended` 缺失 或 `'list' object has no attribute 'keys'` | transformers 版本两头不讨好（v5 太新 / 4.56 遇上 Qwen3 新 tokenizer 配置） | **transformers==4.57.1**；仍不行可将模型目录 `tokenizer_config.json` 的 `extra_special_tokens` 改为 `{}` |
| 4 | pip 下载超时 | PyPI 直连不通 | 清华源 `--index-url` |
| 5 | `externally-managed-environment` / which pip 指向系统 | uv venv 无 pip，PATH 回落 | `uv pip install --python <venv>/bin/python` |
| 6 | `GPU memory utilization` 报错 | Windows 占显存 + 默认 0.92 | 释放 Windows 显存（≤1.5G）+ `--gpu-memory-utilization 0.85` |
| 7 | import/启动奇慢 | venv 或模型在 /mnt/* | 全部放 WSL 原生目录（~/） |
| 8 | 启动久无输出 | torch.compile + CUDA graph 捕获 | 正常，等 `Application startup complete` |
| 9 | 首批请求慢 | 预热懒加载 | 正常，冒烟跑 5 条即热 |
| 10 | model not found（调 API 时） | served name 是完整路径 | `--model` 用路径，或 `--served-model-name` 起短名 |
| 11 | `Failed to find C compiler`（torch.compile/triton 阶段） | 裸 WSL 没有 gcc | `sudo apt install -y build-essential` |
| 12 | `extra_special_tokens` list 报 `'list' object has no attribute 'keys'` | Qwen3 新版 tokenizer 配置与 4.5x 不兼容 | 用 Python 把 `tokenizer_config.json` 的该字段改为 `{}`（备份后改） |
