#!/usr/bin/env python3
import argparse
import copy as cp
import json
import os
import re
import string
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from media_input_utils import add_media_mode_args, build_image_reference, curl_json_request, local_http_media_server, resolve_model_id


MMB_ABBRS = {
    "coarse_perception": "CP",
    "finegrained_perception (instance-level)": "FP-S",
    "finegrained_perception (cross-instance)": "FP-C",
    "logic_reasoning": "LR",
    "relation_reasoning": "RR",
    "attribute_reasoning": "AR",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run MMBench_DEV_EN against a local OpenAI-compatible endpoint.")
    parser.add_argument("--tsv", default="/tmp/MMBench_DEV_EN.tsv", help="Path to official MMBench_DEV_EN.tsv")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default="/mnt/sfs_turbo/models/Qwen/Qwen3.5-4B")
    parser.add_argument("--api-key", default="sk-admin")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for debugging")
    parser.add_argument(
        "--out-prefix",
        default="/mnt/sfs_turbo/codes/lzp/vllm-ascend-precision-testing/mmbench_runs/mmbench_dev_en_qwen35_4b",
        help="Prefix for prediction/result files",
    )
    add_media_mode_args(parser)
    return parser.parse_args()


def can_infer_option(answer, choices):
    if "Failed to obtain answer via API" in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        "Cannot determine the answer",
    ]
    for err in reject_to_answer:
        if err in answer:
            return "Z"

    def count_choice(splits, choices, prefix="", suffix=""):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = cp.copy(str(answer))
    chars = ".()[],:;!*#{}"
    for c in chars:
        answer_mod = answer_mod.replace(c, " ")

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if ch in splits and splits.index(ch) > (len(splits) - 5):
                return ch
    elif count == 0 and count_choice(splits, {"Z", ""}) == 1:
        return "Z"
    return False


def can_infer_text(answer, choices):
    answer = str(answer).lower()
    if len(answer) > 2 * sum(len(str(v)) for v in choices.values()):
        return False
    lowered = {k: str(v).lower() for k, v in choices.items()}
    cands = []
    for k in lowered:
        if lowered[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False


def can_infer(answer, choices):
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)


def report_acc(df):
    res = defaultdict(list)
    if "split" in df:
        splits = list(set(df["split"]))
        res["split"] = splits
    else:
        df["split"] = ["none"] * len(df)
        res["split"] = ["none"]

    for group in [None, "l2-category", "category"]:
        if group is None:
            res["Overall"] = [np.mean(df[df["split"] == sp]["hit"]) for sp in res["split"]]
        elif group not in df:
            continue
        else:
            abilities = list(set(df[group]))
            abilities.sort()
            for ab in abilities:
                ab_name = MMB_ABBRS[ab] if ab in MMB_ABBRS else ab
                sub_df = df[df[group] == ab]
                res[ab_name] = [np.mean(sub_df[sub_df["split"] == sp]["hit"]) for sp in res["split"]]
    return pd.DataFrame(res)


class Client:
    def __init__(self, endpoint, model, api_key, max_tokens, temperature, timeout):
        self.endpoint = endpoint
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    def infer(self, prompt, image_ref):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer directly with only the single uppercase option letter "
                        "from the given choices, such as A, B, C, or D."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_ref}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        last_err = None
        for _ in range(3):
            try:
                status, data, body = curl_json_request(self.endpoint, payload, self.timeout, "POST")
                if status >= 400 or data is None:
                    raise RuntimeError(f"HTTP {status}: {body[:1000]}")
                return data["choices"][0]["message"]["content"].strip()
            except Exception as err:
                last_err = err
        raise last_err


def build_prompt(row):
    parts = []
    hint = row.get("hint")
    if pd.notna(hint):
        parts.append(f"Hint: {hint}")
    parts.append(f"Question: {row['question']}")

    option_lines = []
    for key in string.ascii_uppercase:
        if key in row and pd.notna(row[key]):
            option_lines.append(f"{key}. {row[key]}")
    if option_lines:
        parts.append("Options:\n" + "\n".join(option_lines))
        parts.append("Select the correct answer from the options above.")
    return "\n".join(parts)


def build_choices(row):
    ret = {}
    for ch in string.ascii_uppercase:
        if ch in row and pd.notna(row[ch]):
            ret[ch] = row[ch]
    return ret


def load_rows(tsv_path, limit=0):
    df = pd.read_csv(tsv_path, sep="\t")
    image_map_by_index = {}
    rows = []
    for _, row in df.iterrows():
        item = row.to_dict()
        img = str(item["image"])
        idx = int(item["index"])
        cache_relpath = f"mmbench/{idx}.jpg"
        if len(img) > 100:
            image_map_by_index[idx] = {
                "image_b64": img,
                "cache_relpath": cache_relpath,
            }
            item["image_b64"] = img
            item["cache_relpath"] = cache_relpath
        else:
            ref_idx = int(img)
            if ref_idx not in image_map_by_index:
                raise ValueError(f"Missing referenced image base64 for index={idx}, ref={ref_idx}")
            item["image_b64"] = image_map_by_index[ref_idx]["image_b64"]
            item["cache_relpath"] = image_map_by_index[ref_idx]["cache_relpath"]
        rows.append(item)
        if limit and len(rows) >= limit:
            break
    return rows


def main():
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    endpoint_root = args.endpoint.rsplit("/v1/chat/completions", 1)[0]
    status, models_json, _ = curl_json_request(f"{endpoint_root}/v1/models", None, min(args.timeout, 30), "GET")
    resolved_model = args.model
    if status == 200 and models_json and isinstance(models_json.get("data"), list):
        available_ids = {str(item.get("id", "")) for item in models_json["data"]}
        resolved_model = resolve_model_id(args.model, available_ids)

    rows = load_rows(args.tsv, args.limit)
    media_root = args.media_root or str(out_prefix.parent / "_media_cache" / "mmbench")

    def run_one(i, row):
        prompt = build_prompt(row)
        try:
            prediction = client.infer(prompt, row["image_ref"])
            choices = build_choices(row)
            extracted = can_infer(prediction, choices) or "Z"
            answer = str(row["answer"]).strip().upper()
            row_hit = extracted == answer
            if extracted == "Z":
                failure_class = "output_format_or_extraction_issue"
                root_cause_note = "Model did not reliably collapse to a single choice letter."
            elif row_hit:
                failure_class = "none"
                root_cause_note = "Prediction matched the expected choice."
            else:
                failure_class = "model_capability_gap"
                root_cause_note = "Request completed and decoded, but the predicted choice was wrong."
            return i, {
                "index": int(row["index"]),
                "g_index": int(int(row["index"]) % 1e6),
                "image_path": row["cache_relpath"],
                "media_mode": args.media_mode,
                "image_ref": row["image_ref"],
                "local_image_path": row["local_image_path"],
                "question": row["question"],
                "answer": answer,
                "prediction": prediction,
                "extracted": extracted,
                "row_hit": row_hit,
                "category": row.get("category"),
                "l2-category": row.get("l2-category"),
                "split": row.get("split"),
                "failure_class": failure_class,
                "root_cause_note": root_cause_note,
            }
        except Exception as exc:  # noqa: BLE001
            answer = str(row["answer"]).strip().upper()
            return i, {
                "index": int(row["index"]),
                "g_index": int(int(row["index"]) % 1e6),
                "image_path": row["cache_relpath"],
                "media_mode": args.media_mode,
                "image_ref": row["image_ref"],
                "local_image_path": row["local_image_path"],
                "question": row["question"],
                "answer": answer,
                "prediction": repr(exc),
                "extracted": "Z",
                "row_hit": False,
                "category": row.get("category"),
                "l2-category": row.get("l2-category"),
                "split": row.get("split"),
                "failure_class": "pipeline_or_serving_issue",
                "root_cause_note": "Request failed before a stable choice answer was obtained; treat as serving or transport issue first.",
            }

    with local_http_media_server(args.media_mode, media_root, args.media_base_url) as effective_media_base_url:
        for row in rows:
            image_ref, local_image_path = build_image_reference(
                image_b64=row["image_b64"],
                image_path=row["cache_relpath"],
                media_mode=args.media_mode,
                media_root=media_root,
                media_base_url=effective_media_base_url or "",
                fallback_name=f"mmbench/{int(row['index'])}.jpg",
            )
            row["image_ref"] = image_ref
            row["local_image_path"] = local_image_path

        client = Client(
            endpoint=args.endpoint,
            model=resolved_model,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
        )

        results = [None] * len(rows)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(run_one, i, row) for i, row in enumerate(rows)]
            done = 0
            for fut in as_completed(futures):
                idx, item = fut.result()
                results[idx] = item
                done += 1
                if done % 50 == 0 or done == len(rows):
                    print(f"progress {done}/{len(rows)}", flush=True)

    all_pred_df = pd.DataFrame(results).sort_values(by="index")
    group_hit = all_pred_df.groupby("g_index")["row_hit"].all().to_dict()
    pred_df = all_pred_df[all_pred_df["index"] == all_pred_df["g_index"]].copy()
    pred_df["hit"] = pred_df["g_index"].map(group_hit).astype(bool)
    acc_df = report_acc(pred_df)

    pred_all_path = out_prefix.with_suffix(".pred_all.tsv")
    pred_path = out_prefix.with_suffix(".pred.tsv")
    score_path = out_prefix.with_suffix(".acc.csv")
    all_pred_df.to_csv(pred_all_path, sep="\t", index=False)
    pred_df.to_csv(pred_path, sep="\t", index=False)
    acc_df.to_csv(score_path, index=False)

    summary = {
        "rows_all": len(all_pred_df),
        "rows_scored": len(pred_df),
        "media_mode": args.media_mode,
        "resolved_model": resolved_model,
        "media_root": media_root if args.media_mode != "base64" else "",
        "media_base_url": effective_media_base_url if args.media_mode == "http" else args.media_base_url,
        "exact_acc": float(pred_df["hit"].mean() * 100),
        "z_fallback": int((all_pred_df["extracted"] == "Z").sum()),
        "failure_class_counts": {
            "pipeline_or_serving_issue": int((all_pred_df["failure_class"] == "pipeline_or_serving_issue").sum()),
            "model_capability_gap": int((all_pred_df["failure_class"] == "model_capability_gap").sum()),
            "output_format_or_extraction_issue": int((all_pred_df["failure_class"] == "output_format_or_extraction_issue").sum()),
            "none": int((all_pred_df["failure_class"] == "none").sum()),
        },
        "engineering_error_cases": all_pred_df.loc[all_pred_df["failure_class"] == "pipeline_or_serving_issue", "index"].astype(str).tolist(),
        "model_limitation_cases": all_pred_df.loc[all_pred_df["failure_class"] == "model_capability_gap", "index"].astype(str).tolist(),
        "format_or_protocol_issue_cases": all_pred_df.loc[all_pred_df["failure_class"] == "output_format_or_extraction_issue", "index"].astype(str).tolist(),
        "pred_all_path": str(pred_all_path),
        "pred_path": str(pred_path),
        "score_path": str(score_path),
        "scores": acc_df.iloc[0].to_dict(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("VERBOSE", "0")
    main()
