#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
import urllib.error
import urllib.request

from media_input_utils import build_media_reference


def build_image_part(media_ref: str) -> dict:
    return {"type": "image_url", "image_url": {"url": media_ref}}


def build_video_part(media_ref: str) -> dict:
    return {"type": "video_url", "video_url": {"url": media_ref}}


def build_cases(image_refs: list[tuple[str, str]], video_ref: str):
    imgs = [
        ("circle", image_refs[0][0]),
        ("cube", image_refs[1][0]),
        ("cylinder", image_refs[2][0]),
        ("rectangle", image_refs[3][0]),
        ("rhombus", image_refs[4][0]),
        ("square", image_refs[5][0]),
        ("triangle", image_refs[6][0]),
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
                build_image_part(imgs[0][1]),
            ],
        },
        {
            "id": "img-2-second",
            "expected": "cube",
            "system": system_word,
            "user": [
                {"type": "text", "text": "There are 2 images. What shape is in image 2?"},
                *(build_image_part(p) for _, p in imgs[:2]),
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
                *(build_image_part(p) for _, p in imgs[:3]),
            ],
        },
        {
            "id": "img-4-shape4",
            "expected": "rectangle",
            "system": system_word,
            "user": [
                {"type": "text", "text": "There are 4 images. What shape is in image 4?"},
                *(build_image_part(p) for _, p in imgs[:4]),
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
                *(build_image_part(p) for _, p in imgs[:5]),
            ],
        },
        {
            "id": "img-6-shape6",
            "expected": "square",
            "system": system_word,
            "user": [
                {"type": "text", "text": "There are 6 images. What shape is in image 6?"},
                *(build_image_part(p) for _, p in imgs[:6]),
            ],
        },
        {
            "id": "img-7-last",
            "expected": "triangle",
            "system": system_word,
            "user": [
                {"type": "text", "text": "There are 7 images. What shape is in image 7?"},
                *(build_image_part(p) for _, p in imgs[:7]),
            ],
        },
        {
            "id": "video-first",
            "expected": "square",
            "system": system_word,
            "user": [
                {"type": "text", "text": "What is the first shape that appears in the video?"},
                build_video_part(video_ref),
            ],
        },
        {
            "id": "video-last",
            "expected": "cube",
            "system": system_word,
            "user": [
                {"type": "text", "text": "What is the last shape that appears in the video?"},
                build_video_part(video_ref),
            ],
        },
        {
            "id": "video-count",
            "expected": "7",
            "system": system_num,
            "user": [
                {"type": "text", "text": "How many different shapes appear in the video?"},
                build_video_part(video_ref),
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
    rec = {
        "id": case["id"],
        "expected": case["expected"],
        "returncode": 0,
        "stderr": "",
    }
    request = urllib.request.Request(
        url=f"{base_url}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        finish_reason = data.get("choices", [{}])[0].get("finish_reason")
        rec["content"] = content
        rec["finish_reason"] = finish_reason
        rec["pass"] = content.strip().lower() == case["expected"]
    except urllib.error.HTTPError as exc:
        rec["returncode"] = exc.code
        rec["stderr"] = exc.read().decode("utf-8", errors="replace")
        rec["pass"] = False
    except Exception as exc:  # noqa: BLE001
        rec["returncode"] = 1
        rec["stderr"] = repr(exc)
        rec["pass"] = False
    return rec


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the verified multimodal L0 smoke suite for Qwen3.5/Qwen3-VL style services."
    )
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="/mnt/sfs_turbo/models/Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--image-dir",
        default="/mnt/sfs_turbo/codes/lzp/vllm_multimodal_evaluator/pics/720x1280/jpg",
    )
    parser.add_argument(
        "--video-path",
        default="/mnt/sfs_turbo/codes/lzp/vllm_multimodal_evaluator/video/720x1280/mp4/shapes.mp4",
    )
    parser.add_argument(
        "--media-mode",
        choices=["base64", "local_path", "http"],
        default="local_path",
        help="How to send images and video to the endpoint.",
    )
    parser.add_argument(
        "--media-base-url",
        default="",
        help="Base URL for local HTTP media serving, for example http://127.0.0.1:9000 .",
    )
    parser.add_argument(
        "--media-root",
        default="",
        help="Local media root for HTTP path mapping. Required for http mode.",
    )
    parser.add_argument("--max-completion-tokens", type=int, default=64)
    parser.add_argument("--json", action="store_true", help="Print only JSON results.")
    args = parser.parse_args()

    image_paths = [
        str(Path(args.image_dir) / "circle.jpg"),
        str(Path(args.image_dir) / "cube.jpg"),
        str(Path(args.image_dir) / "cylinder.jpg"),
        str(Path(args.image_dir) / "rectangle.jpg"),
        str(Path(args.image_dir) / "rhombus.jpg"),
        str(Path(args.image_dir) / "square.jpg"),
        str(Path(args.image_dir) / "triangle.jpg"),
    ]
    image_refs = [
        build_media_reference(
            encoded_payload=None,
            source_path=image_path,
            media_mode=args.media_mode,
            media_root=args.media_root,
            media_base_url=args.media_base_url,
            fallback_name=Path(image_path).name,
            fallback_suffix=".jpg",
            fallback_mime="image/jpeg",
        )
        for image_path in image_paths
    ]
    video_ref, video_local_path = build_media_reference(
        encoded_payload=None,
        source_path=args.video_path,
        media_mode=args.media_mode,
        media_root=args.media_root,
        media_base_url=args.media_base_url,
        fallback_name=Path(args.video_path).name,
        fallback_suffix=".mp4",
        fallback_mime="video/mp4",
    )
    cases = build_cases(image_refs, video_ref)
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
                    "summary": {
                        "passed": passed,
                        "failed": failed,
                        "total": len(results),
                        "media_mode": args.media_mode,
                        "video_local_path": video_local_path,
                    },
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
