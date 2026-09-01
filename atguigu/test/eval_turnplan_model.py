"""旺店通 TurnPlanner 锁定集评测。

默认使用 200 条 case spec。case 不保存渲染后的旧 prompt，而是在每次评测时根据当前：
- flow_config/user_flows.yml
- flow_config/system_flows.yml
- KNOWLEDGE_INTENTS
- turn_plan.jinja2
动态渲染，避免业务配置更新后继续评测旧 prompt。

用法：
    # 只验证 200 条 case 能否按当前代码渲染，且金标能通过预期 Validator 行为
    uv run python -m atguigu.test.eval_turnplan_model --render-only

    # 打当前 vLLM TurnPlanner
    uv run python -m atguigu.test.eval_turnplan_model

    # 冒烟
    uv run python -m atguigu.test.eval_turnplan_model --limit 10
"""

import argparse
import asyncio
import json
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate

from atguigu.config.config import settings
from atguigu.domain.messages import MessageType, UserMessage
from atguigu.graph.context import TurnContext
from atguigu.knowledge.intents import KNOWLEDGE_INTENTS
from atguigu.plan.planner import TurnPlanner
from atguigu.plan.turn_plan import TurnPlan
from atguigu.plan.validator import TurnPlanValidator
from atguigu.prompt.loader import load_prompt_template
from atguigu.task.flows.loader import FlowLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL = PROJECT_ROOT / "data" / "turnplan" / "wangdiantong_locked_eval_200.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "data" / "turnplan" / "wangdiantong_model_eval_200.jsonl"
DEFAULT_RENDERED = PROJECT_ROOT / "data" / "turnplan" / "wangdiantong_locked_rendered_200.jsonl"

FLOW_CONFIG_DIR = PROJECT_ROOT / "flow_config"
FLOWS_LIST = FlowLoader().load_multi_yaml([
    FLOW_CONFIG_DIR / "system_flows.yml",
    FLOW_CONFIG_DIR / "user_flows.yml",
])
VALIDATOR = TurnPlanValidator()
PLANNER = TurnPlanner()
TURN_PLAN_TEMPLATE = PromptTemplate.from_template(
    template=load_prompt_template("turn_plan"),
    template_format="jinja2",
)


def _normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_normalize(item) for item in obj]
    return obj


def _history_messages(entries: list[dict[str, Any]]):
    messages = []
    for entry in entries:
        role = entry.get("role")
        text = str(entry.get("text") or "")
        if role == "user":
            messages.append(HumanMessage(content=text))
        else:
            messages.append(AIMessage(content=text))
    return messages


def render_case_prompt(sample: dict[str, Any]) -> str:
    """按当前分支真实配置渲染 prompt；兼容旧 eval_set 中已固化的 prompt。"""
    if isinstance(sample.get("prompt"), str):
        return sample["prompt"]

    runtime = sample.get("state") or {}
    slots = dict(runtime.get("slots") or {})
    focused_object = runtime.get("focused_object")

    user_message = UserMessage(
        sender_id="locked-eval-user",
        message_id=sample["id"],
        type=MessageType.TEXT,
        text=sample["user_message"],
    )
    ctx = TurnContext(
        user_message=user_message,
        history_messages=_history_messages(runtime.get("history") or []),
        slots=slots,
        flow_context=runtime.get("flow_context"),
        focused_object=focused_object,
    )

    prompt_inputs = PLANNER._build_prompt_inputs(
        ctx,
        FLOWS_LIST,
        KNOWLEDGE_INTENTS,
        active_flow=runtime.get("active_flow"),
        active_flow_step=runtime.get("active_flow_step"),
        slots=slots,
        paused_flows=runtime.get("paused_flows") or {},
    )
    return TURN_PLAN_TEMPLATE.format(**prompt_inputs)


def strict_parse(content: str) -> dict[str, Any]:
    """严格格式口径：输出必须是裸 JSON，不接受 markdown fence / <think> / 额外文本。"""
    parsed = json.loads(content.strip())
    if not isinstance(parsed, dict):
        raise ValueError("top-level output must be a JSON object")

    expected_keys = {"task", "knowledge", "chitchat"}
    if set(parsed.keys()) != expected_keys:
        raise ValueError(
            f"top-level keys must be exactly {sorted(expected_keys)}, got {sorted(parsed.keys())}"
        )

    task = parsed["task"]
    knowledge = parsed["knowledge"]
    chitchat = parsed["chitchat"]

    if task is not None:
        if not isinstance(task, dict) or not isinstance(task.get("commands"), list):
            raise ValueError("task.commands must be a list")
        for command in task["commands"]:
            if not isinstance(command, dict) or not isinstance(command.get("command"), str):
                raise ValueError("every task command must be an object with string command")
            if command.get("command") == "set_slots" and not isinstance(command.get("slots"), dict):
                raise ValueError("set_slots.slots must be an object")

    if knowledge is not None:
        if not isinstance(knowledge, dict) or not isinstance(knowledge.get("intents"), list):
            raise ValueError("knowledge.intents must be a list")

    if chitchat is not None:
        if not isinstance(chitchat, dict) or not isinstance(chitchat.get("chat"), str):
            raise ValueError("chitchat.chat must be a string")

    return parsed


def validator_result(plan_dict: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    runtime = sample.get("state") or {}
    result = VALIDATOR.validate(
        TurnPlan.from_dict(plan_dict),
        runtime.get("focused_object"),
        FLOWS_LIST,
        KNOWLEDGE_INTENTS,
        active_flow=runtime.get("active_flow"),
    )
    return {
        "valid": result.valid,
        "reason": result.reason.value if result.reason else None,
    }


def validate_locked_labels(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """先验证金标自身能按当前代码解析/渲染；这一步不调用模型。"""
    rendered = []
    for sample in samples:
        prompt = render_case_prompt(sample)
        expected_validation = validator_result(sample["label"], sample)
        rendered.append({
            "id": sample["id"],
            "group": sample.get("group"),
            "user_message": sample.get("user_message"),
            "prompt": prompt,
            "label": sample["label"],
            "expected_validation": expected_validation,
        })
    return rendered


async def call_vllm(
    base_url: str,
    model: str,
    prompt: str,
    sem: asyncio.Semaphore,
    retries: int = 2,
    api_key: str | None = None,
) -> str:
    """调用 OpenAI 兼容 vLLM 端点。Qwen2.5 不使用 Qwen3 thinking 参数。"""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 300,
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            async with sem:
                def _post():
                    req = urllib.request.Request(
                        f"{base_url.rstrip('/')}/chat/completions",
                        data=body,
                        headers=headers,
                    )
                    return json.loads(urllib.request.urlopen(req, timeout=180).read())

                resp = await asyncio.to_thread(_post)
            return resp["choices"][0]["message"]["content"]
        except Exception as exc:
            last_err = exc
            await asyncio.sleep(2 * (attempt + 1))

    raise last_err  # type: ignore[misc]


async def main() -> None:
    parser = argparse.ArgumentParser(description="旺店通 TurnPlanner 200 条锁定集评测")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rendered-out", type=Path, default=DEFAULT_RENDERED)
    parser.add_argument("--base-url", type=str, default=settings.llm_router_base_url)
    parser.add_argument("--model", type=str, default=settings.llm_router_model)
    parser.add_argument("--api-key", type=str, default=settings.llm_router_api_key)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="只渲染并验证锁定集，不调用模型",
    )
    args = parser.parse_args()

    samples = [
        json.loads(line)
        for line in args.eval_file.open(encoding="utf-8")
        if line.strip()
    ]
    if args.limit:
        samples = samples[:args.limit]

    rendered = validate_locked_labels(samples)
    args.rendered_out.parent.mkdir(parents=True, exist_ok=True)
    with args.rendered_out.open("w", encoding="utf-8") as f:
        for item in rendered:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    group_counts = Counter(item.get("group") or "(legacy)" for item in samples)
    expected_rejections = Counter(
        item["expected_validation"]["reason"]
        for item in rendered
        if not item["expected_validation"]["valid"]
    )

    print(f"锁定集: {len(samples)} 条，场景组: {len(group_counts)}")
    print("场景分布:")
    for group, n in group_counts.most_common():
        print(f"  {group}: {n}")
    if expected_rejections:
        print("预期 Validator 拒绝:")
        for reason, n in expected_rejections.most_common():
            print(f"  {reason}: {n}")
    print(f"动态渲染结果 → {args.rendered_out}")

    if args.render_only:
        print("render-only 完成：未调用模型。")
        return

    rendered_by_id = {item["id"]: item for item in rendered}
    sem = asyncio.Semaphore(args.concurrency)

    async def _one(sample: dict[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {
            "id": sample["id"],
            "group": sample.get("group"),
            "raw": None,
            "parsed": None,
            "format_ok": False,
            "actual_validation": None,
            "expected_validation": rendered_by_id[sample["id"]]["expected_validation"],
            "validation_match": False,
            "match_label": False,
            "error": None,
        }

        try:
            prompt = rendered_by_id[sample["id"]]["prompt"]
            raw = await call_vllm(
                args.base_url,
                args.model,
                prompt,
                sem,
                api_key=args.api_key,
            )
            record["raw"] = raw

            parsed = strict_parse(raw)
            record["parsed"] = parsed
            record["format_ok"] = True

            actual_validation = validator_result(parsed, sample)
            record["actual_validation"] = actual_validation
            record["validation_match"] = (
                actual_validation == record["expected_validation"]
            )
            record["match_label"] = (
                _normalize(parsed) == _normalize(sample["label"])
            )

        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"

        return record

    results = []
    done = 0
    for coro in asyncio.as_completed([_one(sample) for sample in samples]):
        record = await coro
        results.append(record)
        done += 1
        if done % 20 == 0 or done == len(samples):
            print(f"  进度 {done}/{len(samples)}")

    by_id = {record["id"]: record for record in results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(by_id[sample["id"]], ensure_ascii=False) + "\n")

    total = len(samples)
    format_ok = sum(1 for r in results if r["format_ok"])
    semantic_match = sum(1 for r in results if r["match_label"])
    validation_match = sum(1 for r in results if r["validation_match"])

    print("\n========== 旺店通锁定集评测 ==========")
    print(f"严格格式合规:     {format_ok}/{total} = {format_ok / total:.2%}")
    print(f"语义精确一致:     {semantic_match}/{total} = {semantic_match / total:.2%}")
    print(f"Validator行为一致: {validation_match}/{total} = {validation_match / total:.2%}")

    bad = [r for r in results if not r["match_label"]]
    print(f"\n语义不一致 {len(bad)} 条，按场景组分布:")
    for group, n in Counter(r.get("group") or "(legacy)" for r in bad).most_common():
        print(f"  {group}: {n}/{group_counts.get(group, 0)}")

    format_bad = [r for r in results if not r["format_ok"]]
    if format_bad:
        print(f"\n格式错误 {len(format_bad)} 条，前 10 条:")
        for r in format_bad[:10]:
            print(f"  {r['id']}: {r['error']}")

    print(f"\n逐条模型结果 → {args.out}")
    print("注意：旧 100 条 teacher 指标属于旧业务 prompt，不作为本 200 条旺店通锁定集基线。")


if __name__ == "__main__":
    asyncio.run(main())
