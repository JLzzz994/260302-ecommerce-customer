"""
TurnPlanner 微调模型评测脚本：eval 集 100 条 → 打 vLLM 端点 → 四指标 + 分格子错误分布

用法（vLLM 服务需已启动，AutoDL 公网映射 + api-key）：
    uv run python -m atguigu.test.eval_turnplan_model
    uv run python -m atguigu.test.eval_turnplan_model --limit 5          # 冒烟
    uv run python -m atguigu.test.eval_turnplan_model --api-key xxx      # 显式指定 key
输出：
    data/turnplan/model_eval.jsonl   逐条结果（id/parsed/valid/reason/match_label/error）
    控制台四指标汇总 + 按父场景错误分布（与 teacher_eval.jsonl 同口径，可直接对比）
"""
import argparse
import asyncio
import json
import urllib.request
from pathlib import Path
from typing import Any

from atguigu.config.config import settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL = PROJECT_ROOT / "data" / "turnplan" / "eval_set.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "data" / "turnplan" / "model_eval.jsonl"


def _normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_normalize(item) for item in obj]
    return obj


async def call_vllm(base_url: str, model: str, prompt: str, sem: asyncio.Semaphore,
                    retries: int = 2, api_key: str | None = None) -> str:
    """打 OpenAI 兼容端点，返回原始文本（带重试；vLLM 开了 --api-key 时必须带 Bearer）"""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 300,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            async with sem:
                def _post():
                    req = urllib.request.Request(f"{base_url}/chat/completions",
                                                 data=body,
                                                 headers=headers)
                    return json.loads(urllib.request.urlopen(req, timeout=180).read())
                resp = await asyncio.to_thread(_post)
            return resp["choices"][0]["message"]["content"]
        except Exception as exc:
            last_err = exc
            await asyncio.sleep(2 * (attempt + 1))
    raise last_err  # type: ignore[misc]


def parse_content(content: str) -> dict[str, Any]:
    """模型输出 → dict。剥 <think> 块、剥 markdown 围栏，取第一个 JSON 对象"""
    import re
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned).rstrip("`").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in output: {content[:100]!r}")
    return json.loads(cleaned[start:end + 1])


async def main() -> None:
    parser = argparse.ArgumentParser(description="TurnPlanner 模型评测")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base-url", type=str,
                        default="https://INSTANCE-REDACTED.westc.seetacloud.com:8443/v1")
    parser.add_argument("--model", type=str, default="turnplanner")
    parser.add_argument("--api-key", type=str, default=settings.llm_router_api_key,
                        help="vLLM --api-key 的值（默认读 settings.llm_router_api_key）")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    # 1. 读 eval 集（label 已人工核对）
    samples = [json.loads(line) for line in args.eval_file.open(encoding="utf-8")]
    if args.limit:
        samples = samples[:args.limit]

    # 2. 并发打端点
    sem = asyncio.Semaphore(args.concurrency)

    async def _one(sample: dict[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {"id": sample["id"], "parent": sample.get("parent"),
                                  "raw": None, "parsed": None, "valid": None,
                                  "reason": None, "match_label": None, "error": None}
        try:
            content = await call_vllm(args.base_url, args.model, sample["prompt"], sem,
                                      api_key=args.api_key)
            record["raw"] = content
            parsed = parse_content(content)
            record["parsed"] = parsed
            # schema 级结构校验（与 TurnPlan.from_dict 同构的硬解析）
            task = parsed.get("task")
            if task is not None:
                for command in task.get("commands", []):
                    if command.get("command") not in {"start_flow", "resume_flow",
                                                      "cancel_flow", "set_slots"}:
                        raise ValueError(f"invalid command: {command.get('command')}")
            record["valid"] = True
            record["match_label"] = _normalize(parsed) == _normalize(sample["label"])
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["valid"] = False
            record["reason"] = "parse_or_schema_error"
        return record

    done = 0
    results = []
    for coro in asyncio.as_completed([_one(s) for s in samples]):
        record = await coro
        results.append(record)
        done += 1
        if done % 10 == 0 or done == len(samples):
            print(f"  进度 {done}/{len(samples)}")

    # 3. 落盘（按原样本顺序对齐 id，方便和 teacher_eval 逐条 diff）
    by_id = {r["id"]: r for r in results}
    with args.out.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(by_id[sample["id"]], ensure_ascii=False) + "\n")

    # 4. 四指标汇总
    total = len(samples)
    parse_ok = sum(1 for r in results if r["valid"])
    match = sum(1 for r in results if r["match_label"])
    print("\n========== 评测结果 ==========")
    print(f"格式合规率(valid):  {parse_ok}/{total} = {parse_ok / total:.1%}   [teacher 基线 94.0%]")
    print(f"语义一致率(match):   {match}/{total} = {match / total:.1%}   [teacher 基线 71.0%]")

    # 5. 分格子错误分布
    from collections import Counter
    bad = [r for r in results if not r["match_label"]]
    print(f"\n不一致 {len(bad)} 条，按父场景分布:")
    parent_total = Counter(s.get("parent") or s["id"] for s in samples)
    for parent, n in Counter(r.get("parent") or r["id"] for r in bad).most_common():
        print(f"  {parent or '(基础场景)'}: {n}/{parent_total.get(parent, 0)}")
    print(f"\n逐条结果 → {args.out}")
    print("（与 teacher 对比：data/turnplan/teacher_eval.jsonl 同 id 对齐）")


if __name__ == "__main__":
    asyncio.run(main())
