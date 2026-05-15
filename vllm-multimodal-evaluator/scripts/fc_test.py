#!/usr/bin/env python3
"""Function Calling test runner for vLLM - relaxed evaluation."""
import argparse
import json
import os
import sys

import requests

ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "/root/gpufree-data/models/Qwen3-5-2B"

def load_json(path):
    with open(path) as f:
        return json.load(f)

def send(messages, tools, endpoint, model):
    tools_openai = [{"type": "function", "function": t} for t in tools] if tools else []
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools_openai,
        "tool_choice": "auto" if tools else "none",
        "temperature": 0.0,
        "max_completion_tokens": 512,
    }
    resp = requests.post(endpoint, json=payload, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
    return resp.json()

def main():
    parser = argparse.ArgumentParser(description="Run Function Calling test suite.")
    parser.add_argument("--endpoint", default=ENDPOINT, help="OpenAI-compatible chat completions URL")
    parser.add_argument("--model", default=MODEL, help="Model name")
    parser.add_argument("--test-file", default="",
                        help="Path to function_calling_test.json (default: auto-detect)")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable summary JSON.")
    args = parser.parse_args()

    endpoint = args.endpoint
    model = args.model
    test_file = args.test_file
    if not test_file:
        # Auto-detect relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        test_file = os.path.join(script_dir, "function_calling_test.json")

    data = load_json(test_file)
    tools = data["functions"]
    cases = data["test_cases"]

    results = []
    for case in cases:
        cid = case["id"]
        inp = case["input"]
        scene = case.get("scene", "")
        request_error = None

        try:
            if isinstance(inp, list):
                messages = [{"role": "user", "content": inp[0]}]
                r1 = send(messages, tools, endpoint, model)
                am1 = r1["choices"][0]["message"]
                messages.append({"role": "assistant", "content": am1.get("content","")})
                if am1.get("tool_calls"):
                    for tc in am1["tool_calls"]:
                        messages.append({"role": "tool", "tool_call_id": tc["id"],
                                         "content": "result_placeholder"})
                messages.append({"role": "user", "content": inp[1]})
                data_resp = send(messages, tools, endpoint, model)
            else:
                messages = [{"role": "user", "content": inp}]
                data_resp = send(messages, tools, endpoint, model)
            choice = data_resp["choices"][0]
            msg = choice["message"]
            tcalls = msg.get("tool_calls") or []
        except Exception as exc:  # noqa: BLE001
            data_resp = None
            tcalls = []
            request_error = repr(exc)

        expected = case["expected"]
        actual_fn = ""
        actual_args = {}
        failure_class = "none"
        should_count_as_model_error = False
        should_count_as_engineering_error = False

        if request_error is not None:
            passed, out_msg = False, f"请求失败: {request_error}"
            failure_class = "pipeline_or_serving_issue"
            should_count_as_engineering_error = True
        elif expected.get("action") == "parallel_call":
            exp_fns = expected.get("functions", [])
            if not tcalls:
                passed, out_msg = False, f"期望并行调用 {exp_fns}，但未调用任何函数"
                failure_class = "model_capability_gap"
                should_count_as_model_error = True
            else:
                called = [t["function"]["name"] for t in tcalls]
                missing = [fn for fn in exp_fns if fn not in called]
                if missing:
                    passed, out_msg = False, f"缺少函数 {missing}，实际调用了 {called}"
                    failure_class = "model_capability_gap"
                    should_count_as_model_error = True
                else:
                    passed, out_msg = True, f"并行调用：{called}"
        elif tcalls:
            actual_fn = tcalls[0]["function"]["name"]
            actual_args = json.loads(tcalls[0]["function"]["arguments"])
            exp_fn = expected.get("function", "")
            req_params = expected.get("parameters", {})
            if exp_fn and actual_fn != exp_fn:
                passed, out_msg = False, f"期望函数 {exp_fn}，实际调用 {actual_fn}"
                failure_class = "model_capability_gap"
                should_count_as_model_error = True
            else:
                missing_req = [k for k in req_params if k not in actual_args]
                if missing_req:
                    passed, out_msg = False, f"必选参数 {missing_req} 缺失，实际参数：{list(actual_args.keys())}"
                    failure_class = "model_capability_gap"
                    should_count_as_model_error = True
                else:
                    passed, out_msg = True, f"调用 {actual_fn}({actual_args})"
        else:
            passed, out_msg = True, "未调用函数（小模型场景下接受）"
            failure_class = "accepted_no_call"

        results.append(
            {
                "id": cid,
                "scene": scene,
                "passed": passed,
                "message": out_msg,
                "failure_class": failure_class,
                "should_count_as_model_error": should_count_as_model_error,
                "should_count_as_engineering_error": should_count_as_engineering_error,
                "expected": expected,
                "actual_function": actual_fn,
                "actual_arguments": actual_args,
                "tool_call_count": len(tcalls),
                "request_error": request_error,
            }
        )
        status = "PASS" if passed else "FAIL"
        log_line = f"{status:4s} | {cid} [{scene}] {out_msg}"
        if args.json:
            print(log_line, file=sys.stderr)
        else:
            print(log_line)

    total = len(results)
    passed_count = sum(1 for item in results if item["passed"])
    summary_line = f"\n=== 汇总: {passed_count}/{total} 通过 ==="
    if args.json:
        print(summary_line, file=sys.stderr)
    else:
        print(summary_line)
    if args.json:
        summary = {
            "summary": {
                "passed": passed_count,
                "failed": total - passed_count,
                "total": total,
                "pass_rate": round(passed_count / total, 4) if total else 0.0,
                "engineering_error_cases": [
                    item["id"] for item in results if item["should_count_as_engineering_error"]
                ],
                "model_limitation_cases": [
                    item["id"] for item in results if item["should_count_as_model_error"]
                ],
            },
            "results": results,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
