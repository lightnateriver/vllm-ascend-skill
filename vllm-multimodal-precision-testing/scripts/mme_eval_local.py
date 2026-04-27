#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from media_input_utils import add_media_mode_args, build_image_reference


def parse_args():
    parser = argparse.ArgumentParser(description="Run MME against a local OpenAI-compatible endpoint.")
    parser.add_argument("--tsv", default="/tmp/MME.tsv", help="Path to official MME.tsv")
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
        default="/mnt/sfs_turbo/codes/lzp/vllm-ascend-precision-testing/mme_runs/mme_qwen35_4b",
        help="Prefix for prediction/result files",
    )
    add_media_mode_args(parser)
    return parser.parse_args()


def process_punctuation(text):
    return re.sub(r"[^\w\s]", " ", str(text)).lower()


def extract_yes_no(output):
    words = process_punctuation(output).split()
    if "yes" in words and "no" not in words:
        return "Yes"
    if "yes" not in words and "no" in words:
        return "No"
    return "Unknown"


def mme_rating(df):
    stats = defaultdict(dict)
    for _, item in df.iterrows():
        category = item["category"]
        image_path = item["image_path"]
        score = bool(item["score"])
        if image_path not in stats[category]:
            stats[category][image_path] = []
        stats[category][image_path].append(score)

    def acc(key, mode="normal"):
        res = stats[key]
        values = []
        for val in res.values():
            if mode == "normal":
                values.extend(val)
            elif mode == "plus":
                if len(val) >= 2:
                    values.append(val[0] * val[1])
        return sum(values) / len(values) * 100 if values else 0.0

    scores = {}
    for k in stats:
        scores[k] = acc(k) + acc(k, "plus")

    super_cates = {
        "perception": [
            "OCR", "artwork", "celebrity", "color", "count", "existence",
            "landmark", "position", "posters", "scene",
        ],
        "reasoning": ["code_reasoning", "commonsense_reasoning", "numerical_calculation", "text_translation"],
    }

    ret = {}
    for sc, cate_list in super_cates.items():
        ret[sc] = sum(scores.get(c, 0.0) for c in cate_list)
    ret.update(scores)
    return pd.DataFrame([ret])


class Client:
    def __init__(self, endpoint, model, api_key, max_tokens, temperature, timeout):
        self.endpoint = endpoint
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def infer(self, question, image_ref):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": image_ref}},
                    ],
                }
            ],
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        last_err = None
        for _ in range(3):
            try:
                resp = requests.post(
                    self.endpoint,
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=self.timeout,
                )
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:1000]}")
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as err:
                last_err = err
        raise last_err


def load_mme_rows(tsv_path, limit=0):
    df = pd.read_csv(tsv_path, sep="\t")
    image_map = {}
    rows = []
    for _, row in df.iterrows():
        image_path = row["image_path"]
        image = str(row["image"])
        # In MME.tsv, repeated images may be marked with short numeric references
        # like "0", "2", "4" instead of an actual base64 payload.
        if image != "nan" and len(image) > 100:
            image_map[image_path] = image
        if image_path not in image_map:
            raise ValueError(f"Missing base64 for image_path={image_path}")
        rows.append(
            {
                "index": row["index"],
                "category": row["category"],
                "image_path": image_path,
                "question": row["question"],
                "answer": row["answer"],
                "image_b64": image_map[image_path],
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows


def main():
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = load_mme_rows(args.tsv, args.limit)
    media_root = args.media_root or str(out_prefix.parent / "_media_cache" / "mme")
    for row in rows:
        image_ref, local_image_path = build_image_reference(
            image_b64=row["image_b64"],
            image_path=row["image_path"],
            media_mode=args.media_mode,
            media_root=media_root,
            media_base_url=args.media_base_url,
            fallback_name=f"mme/{row['image_path']}",
        )
        row["image_ref"] = image_ref
        row["local_image_path"] = local_image_path

    client = Client(
        endpoint=args.endpoint,
        model=args.model,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
    )

    results = [None] * len(rows)

    def run_one(i, row):
        prediction = client.infer(row["question"], row["image_ref"])
        extracted = extract_yes_no(prediction)
        score = extracted == row["answer"]
        return i, {
            "index": row["index"],
            "category": row["category"],
            "image_path": row["image_path"],
            "media_mode": args.media_mode,
            "image_ref": row["image_ref"],
            "local_image_path": row["local_image_path"],
            "question": row["question"],
            "answer": row["answer"],
            "prediction": prediction,
            "extracted": extracted,
            "score": score,
        }

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run_one, i, row) for i, row in enumerate(rows)]
        done = 0
        for fut in as_completed(futures):
            idx, item = fut.result()
            results[idx] = item
            done += 1
            if done % 50 == 0 or done == len(rows):
                print(f"progress {done}/{len(rows)}", flush=True)

    pred_df = pd.DataFrame(results)
    score_df = mme_rating(pred_df)

    pred_path = out_prefix.with_suffix(".pred.tsv")
    score_path = out_prefix.with_suffix(".score.csv")
    pred_df.to_csv(pred_path, sep="\t", index=False)
    score_df.to_csv(score_path, index=False)

    summary = {
        "rows": len(pred_df),
        "media_mode": args.media_mode,
        "media_root": media_root if args.media_mode != "base64" else "",
        "media_base_url": args.media_base_url,
        "exact_acc": float(pred_df["score"].mean() * 100),
        "unknown": int((pred_df["extracted"] == "Unknown").sum()),
        "pred_path": str(pred_path),
        "score_path": str(score_path),
        "scores": score_df.iloc[0].to_dict(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
