#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
DEFAULT_IMAGE_DIR = SKILL_DIR / "assets" / "l0" / "pics" / "720x1280" / "jpg"
DEFAULT_VIDEO_PATH = SKILL_DIR / "assets" / "l0" / "video" / "720x1280" / "mp4" / "shapes.mp4"


def build_cases(image_dir: str, video_path: str):
    imgs = [
        ("circle", f"{image_dir}/circle.jpg"),
        ("cube", f"{image_dir}/cube.jpg"),
        ("cylinder", f"{image_dir}/cylinder.jpg"),
        ("rectangle", f"{image_dir}/rectangle.jpg"),
        ("rhombus", f"{image_dir}/rhombus.jpg"),
        ("square", f"{image_dir}/square.jpg"),
        ("triangle", f"{image_dir}/triangle.jpg"),
    ]

    system_word = "Reply with exactly one lowercase English word and nothing else."
    system_num = "Reply with exactly one number and nothing else."
    system_list = (
        "Reply with exactly the requested comma-separated lowercase English items, "
        "with no spaces and nothing else."
    )
    system_numlist = (
        "Reply with exactly the requested comma-separated numbers, "
        "with no spaces and nothing else."
    )

    return [
        {
            "id": "img-1-single",
            "expected": "circle",
            "system": system_word,
            "user": [
                {"type": "text", "text": "What shape is in this image?"},
                {"type": "image_url", "image_url": {"url": "file://" + imgs[0][1]}},
            ],
        },
        {
            "id": "img-2-second",
            "expected": "cube",
            "system": system_word,
            "user": [
                {"type": "text", "text": "There are 2 images. What shape is in image 2?"},
                *(
                    {"type": "image_url", "image_url": {"url": "file://" + p}}
                    for _, p in imgs[:2]
                ),
            ],
        },
        {
            "id": "img-3-order",
            "expected": "circle,cube,cylinder",
            "system": system_list,
            "user": [
                {
                    "type": "text",
                    "text": "There are 3 images. Return the shapes in order from image 1 to image 3.",
                },
                *(
                    {"type": "image_url", "image_url": {"url": "file://" + p}}
                    for _, p in imgs[:3]
                ),
            ],
        },
        {
            "id": "img-4-shape4",
            "expected": "rectangle",
            "system": system_word,
            "user": [
                {"type": "text", "text": "There are 4 images. What shape is in image 4?"},
                *(
                    {"type": "image_url", "image_url": {"url": "file://" + p}}
                    for _, p in imgs[:4]
                ),
            ],
        },
        {
            "id": "img-5-circle-rhombus",
            "expected": "1,5",
            "system": system_numlist,
            "user": [
                {
                    "type": "text",
                    "text": "There are 5 images. Give the image index of circle first and rhombus second.",
                },
                *(
                    {"type": "image_url", "image_url": {"url": "file://" + p}}
                    for _, p in imgs[:5]
                ),
            ],
        },
        {
            "id": "img-6-shape6",
            "expected": "square",
            "system": system_word,
            "user": [
                {"type": "text", "text": "There are 6 images. What shape is in image 6?"},
                *(
                    {"type": "image_url", "image_url": {"url": "file://" + p}}
                    for _, p in imgs[:6]
                ),
            ],
        },
        {
            "id": "img-7-last",
            "expected": "triangle",
            "system": system_word,
            "user": [
                {"type": "text", "text": "There are 7 images. What shape is in image 7?"},
                *(
                    {"type": "image_url", "image_url": {"url": "file://" + p}}
                    for _, p in imgs[:7]
                ),
            ],
        },
        {
            "id": "video-first",
            "expected": "square",
            "system": system_word,
            "user": [
                {"type": "text", "text": "What is the first shape that appears in the video?"},
                {"type": "video_url", "video_url": {"url": "file://" + video_path}},
            ],
        },
        {
            "id": "video-last",
            "expected": "cube",
            "system": system_word,
            "user": [
                {"type": "text", "text": "What is the last shape that appears in the video?"},
                {"type": "video_url", "video_url": {"url": "file://" + video_path}},
            ],
        },
        {
            "id": "video-count",
            "expected": "7",
            "system": system_num,
            "user": [
                {"type": "text", "text": "How many different shapes appear in the video?"},
                {"type": "video_url", "video_url": {"url": "file://" + video_path}},
            ],
        },
    ]


def run_case(base_url: str, model: str, case: dict, max_completion_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": case["system"]},
            {"role": "user", "content": case["user"]},
        ],
        "temperature": 0,
        "max_completion_tokens": max_completion_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    proc = subprocess.run(
        [
            "curl",
            "-sS",
            f"{base_url}/v1/chat/completions",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(payload, ensure_ascii=False),
        ],
        capture_output=True,
        text=True,
    )
    rec = {
        "id": case["id"],
        "expected": case["expected"],
        "returncode": proc.returncode,
        "stderr": proc.stderr.strip(),
    }
    if proc.returncode != 0:
        rec["pass"] = False
        return rec

    data = json.loads(proc.stdout)
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    finish_reason = data.get("choices", [{}])[0].get("finish_reason")
    rec["content"] = content
    rec["finish_reason"] = finish_reason
    rec["pass"] = content.strip().lower() == case["expected"]
    return rec


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the verified multimodal L0 smoke suite for Qwen3.5/Qwen3-VL style services."
    )
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="/mnt/sfs_turbo/models/Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--image-dir",
        default=str(DEFAULT_IMAGE_DIR),
    )
    parser.add_argument(
        "--video-path",
        default=str(DEFAULT_VIDEO_PATH),
    )
    parser.add_argument("--max-completion-tokens", type=int, default=64)
    parser.add_argument("--json", action="store_true", help="Print only JSON results.")
    args = parser.parse_args()

    cases = build_cases(args.image_dir, args.video_path)
    results = [
        run_case(args.host, args.model, case, args.max_completion_tokens)
        for case in cases
    ]
    passed = sum(1 for r in results if r["pass"])
    failed = len(results) - passed

    if args.json:
        print(
            json.dumps(
                {
                    "summary": {"passed": passed, "failed": failed, "total": len(results)},
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if failed == 0 else 1

    print(f"L0 multimodal smoke: {passed}/{len(results)} passed")
    for rec in results:
        status = "PASS" if rec["pass"] else "FAIL"
        line = f"[{status}] {rec['id']}: expected={rec['expected']}"
        if "content" in rec:
            line += f" got={rec['content']!r}"
        elif rec["stderr"]:
            line += f" stderr={rec['stderr']!r}"
        print(line)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
