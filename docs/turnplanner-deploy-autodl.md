# TurnPlanner 推理服务部署指南（AutoDL + vLLM LoRA serving）【现行】

> 目标：在 AutoDL GPU 实例上，用 vLLM 部署**基座 Qwen3-1.7B + checkpoint-63 LoRA adapter**（运行时浮点挂载，不合并），经公网 HTTPS 映射对 Windows 侧业务提供 OpenAI 兼容端点。
>
> 取代已废弃的 WSL 部署（`turnplanner-deploy-wsl.md`）。**为什么不用合并权重**：bf16 合并会抹除小量级 LoRA delta（52.8%），详见 `turnplanner-bf16-merge-pitfall.md`。
>
> 验证状态（2026-08-20）：此部署评测 **100% 格式合规 / 99% 语义一致**（teacher 基线 94%/71%）。

---

## 0. 架构与资产位置

| 组件 | 位置 | 说明 |
|---|---|---|
| vLLM venv | `/root/autodl-tmp/turnplan-env` | vllm 0.11.0 + transformers 4.57.1 + torch 2.8.0+cu128 + 全套 nvidia cu128 库 |
| wheel 仓库 | `/root/autodl-tmp/wheels*`（~5.6G） | 离线重装/克隆实例用 |
| 基座模型 | `/root/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master` | 系统盘（克隆实例注意：AutoDL 数据盘克隆不带系统盘内容） |
| **adapter（真身）** | `/root/autodl-tmp/output/full/v0-20260819-233903/checkpoint-63` | 本地归档 `D:\models\turnplanner-adapter-v1\`（回放 5/5 验证） |
| 服务端口 | 容器内 127.0.0.1:6006 | 经 AutoDL 公网 HTTPS 映射对外 |
| 公网地址 | `https://<实例映射域名>:8443/v1` | **实例重启会变**，变了改 `.env` 一行 |

## 1. 启动命令（验证过的完整版）

```bash
/root/autodl-tmp/turnplan-env/bin/vllm serve \
  /root/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master \
  --host 127.0.0.1 \
  --port 6006 --max-model-len 4096 \
  --enable-lora --max-lora-rank 16 \
  --served-model-name qwen3-1.7b \
  --api-key sk-REDACTED \
  --dtype auto \
  --trust-remote-code \
  --lora-modules turnplanner=/root/autodl-tmp/output/full/v0-20260819-233903/checkpoint-63
```

要点：
- **业务路由请求 `model` 字段用 `turnplanner`**（LoRA 别名），不是 `qwen3-1.7b`；
- `--max-lora-rank 16` 按 adapter 实际 rank 精确预留，不浪费显存；
- `--api-key` 开了鉴权，所有客户端必须带 `Authorization: Bearer sk-REDACTED`。

## 2. Windows 侧接入（已完成，git 已提交）

| 改动 | 文件 | 内容 |
|---|---|---|
| 配置 | `atguigu/config/config.py` + `.env` | `LLM_ROUTER_MODEL/BASE_URL/API_KEY` 三变量 |
| 双通道 | `atguigu/infrastructure/llm.py` | `llm_client`（表达层大模型）/ `router_client`（路由层微调模型，固定 `enable_thinking=False` 与训练渲染对齐） |
| 切换 | `atguigu/plan/planner.py` | 链中 `llm_client` → `router_client` |
| 评测 | `atguigu/test/eval_turnplan_model.py` | 已带 Bearer 鉴权，`--base-url/--model` 默认即本部署 |

回滚：`.env` 把 `LLM_ROUTER_*` 三行改成 `LLM_*` 同值，重启生效，代码零改动。

## 3. 评测（Windows 侧）

```powershell
# 冒烟
uv run python -m atguigu.test.eval_turnplan_model --limit 5
# 全量 100 条
uv run python -m atguigu.test.eval_turnplan_model
```

基线对照：teacher 94%/71%；本部署实测 100%/99%（`data/turnplan/model_eval.jsonl` 同 id 可 diff）。

## 4. 换卡/克隆实例后的环境恢复

数据盘内容（venv/wheels/adapter）随克隆保留，系统盘内容（miniconda、基座 modelscope 缓存）**不保留**：

```bash
# 1) 基座恢复（modelscope 自动下载到系统盘）
/root/miniconda3/bin/pip download -d /tmp/base modelscope  # 若 miniconda 也需重装则先装它
# 直接确认缓存是否存在：
ls /root/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master/*.safetensors

# 2) venv 即用（数据盘已带）
/root/autodl-tmp/turnplan-env/bin/python -c "import vllm; print(vllm.__version__)"

# 3) 启动命令同第 1 节
```

若基座丢失，从本地 `D:\models\Qwen3-1.7B-base\` 上传，或 modelscope 重新下载。

## 5. 坑清单（本次部署实测，WSL 版 12 坑之外的新增）

| # | 症状 | 根因 | 解法 |
|---|---|---|---|
| 1 | 服务器 pip 下载仅 7KB/s | 实例对外限速 ~0.3MB/s | 大 wheel 用「Windows 下载 → SFTP 上传」；或服务器侧 pip 走 aliyun 源（实测通畅，与 curl 直连不同路） |
| 2 | Windows `pip download` 缺 nvidia/triton/xformers 等 18 个包 | `platform_system=="Linux"` 条件依赖在 Windows 下载机被静默跳过 | 服务器侧 `pip download --no-deps <pkg>` 补齐；cudnn 9.10.2.21 只在 aliyun/pytorch.org 索引有 |
| 3 | 装出 torch-2.8.0+**cpu** | wheels-cpu 目录的 CPU 版 torch 抢位 | `mv` 走 CPU 版 wheel + `pip install --force-reinstall --no-deps <cuda版.whl>` |
| 4 | `libcudnn.so.9: cannot open shared object` | nvidia 库不随 CPU torch 安装 | 显式 `pip install` 全部 14 个 `nvidia-*-cu12` 精确版本 |
| 5 | 装完缺 uvloop 等零星小包 | 同坑 2 的平台条件依赖（uvicorn[standard]） | 自愈循环：`--no-index` 装 → 解析报错里的缺包名 → `pip download` 补 → 重试 |
| 6 | vLLM 起 API 后所有请求 401 | 启动带了 `--api-key` 但客户端没带 Bearer | 客户端统一 `Authorization: Bearer sk-REDACTED`（评测脚本已修） |
| 7 | SSH 连不上（banner 空）但推理服务正常 | 实例 SSH 服务挂/被回收 | AutoDL 控制台重启实例；**注意公网映射地址会变** |
| 8 | 无卡模式跑模型实验 OOM（exit 137） | 无卡容器内存限额 2GB | 实验放本地（模型 sha256 双端一致则结果等价） |

## 6. 成本与生命周期提醒

- GPU 按小时计费（4090 ≈2 元/时）：**评测/联调时开机，闲时关机**；
- 公网 HTTPS 映射地址实例级绑定：**每次重启实例后更新 `.env` 的 `LLM_ROUTER_BASE_URL`**；
- 长期挂业务：把 vLLM 挪到自有机器，本地归档 `D:\models\turnplanner-adapter-v1\` 直接 `--lora-modules turnplanner=<目录>` 复活；
- adapter 迭代（v2）：补数据 → AutoDL 重训 → `swift export` 不做 → 直接换 `--lora-modules` 指向新 checkpoint。
