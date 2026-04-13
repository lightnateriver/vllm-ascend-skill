#!/usr/bin/env python3

import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a cache-safe multimodal dataset with unique text and image "
            "content for vLLM/vllm-ascend service benchmarking."
        )
    )
    parser.add_argument("--tokenizer", required=True, help="Model or tokenizer path")
    parser.add_argument("--request-model", required=True, help="Model field to write into payloads")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=13)
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument("--images-per-round", type=int, default=40)
    parser.add_argument("--width", type=int, default=288)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--target-text-tokens", type=int, default=10000)
    parser.add_argument("--max-completion-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260413)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.rounds <= 0:
        raise ValueError("--rounds must be greater than 0")
    if args.warmup_rounds < 0 or args.warmup_rounds >= args.rounds:
        raise ValueError("--warmup-rounds must be in [0, rounds)")
    if args.images_per_round <= 0:
        raise ValueError("--images-per-round must be greater than 0")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be greater than 0")
    if args.target_text_tokens <= 0:
        raise ValueError("--target-text-tokens must be greater than 0")


def build_sentence(round_id: int, sentence_id: int) -> str:
    code = f"r{round_id:02d}_s{sentence_id:05d}"
    adjectives = [
        "granular",
        "deterministic",
        "cacheless",
        "multimodal",
        "latency-aware",
        "token-precise",
        "vision-heavy",
        "nonrepeating",
    ]
    nouns = [
        "benchmark",
        "request",
        "payload",
        "profile",
        "timeline",
        "dataset",
        "session",
        "sample",
    ]
    adj = adjectives[sentence_id % len(adjectives)]
    noun = nouns[(sentence_id * 3) % len(nouns)]
    numbers = [str(round_id), str(sentence_id), str(round_id * 1000 + sentence_id)]
    return (
        f"Segment {code} records a {adj} {noun} for service tracing. "
        f"It includes identifiers {'/'.join(numbers)} and unique markers "
        f"{code.upper()}::{sentence_id * 17 + round_id}. "
        f"This line is intentionally unique within and across rounds.\n"
    )


def fit_tail_exact(
    tokenizer: AutoTokenizer,
    prefix: str,
    round_id: int,
    sentence_id: int,
    target_tokens: int,
) -> str:
    current = len(tokenizer.encode(prefix, add_special_tokens=False))
    if current == target_tokens:
        return prefix

    def candidate_texts(seed: int) -> list[str]:
        return [
            f" tail{round_id:02d}_{sentence_id:05d}_{seed:05d}",
            f" item-{round_id:02d}-{sentence_id:05d}-{seed:05d}",
            f" note[{round_id:02d}:{sentence_id:05d}:{seed:05d}]",
            f" ref{seed:05d}",
            f" z{seed:05d}",
            f"\nextra_{round_id:02d}_{sentence_id:05d}_{seed:05d}",
            f" code={round_id * 100000 + sentence_id * 97 + seed}",
            f" flag{seed:05d}.",
        ]

    remaining = target_tokens - current
    stack: list[tuple[str, int, int]] = [(prefix, remaining, 1)]
    text = None

    while stack:
        base_text, base_remaining, seed = stack.pop()
        if base_remaining == 0:
            text = base_text
            break
        if base_remaining < 0 or seed > 4096:
            continue

        current_tokens = len(tokenizer.encode(base_text, add_special_tokens=False))
        options: list[tuple[int, str]] = []
        for fragment in candidate_texts(seed):
            merged = base_text + fragment
            merged_tokens = len(tokenizer.encode(merged, add_special_tokens=False))
            delta = merged_tokens - current_tokens
            if 0 < delta <= base_remaining:
                options.append((delta, merged))

        options.sort(key=lambda item: (-item[0], len(item[1])))
        stack.append((base_text, base_remaining, seed + 1))
        for delta, merged in reversed(options):
            stack.append((merged, base_remaining - delta, seed + 1))

    if text is None:
        raise RuntimeError(
            f"failed to fit exact token count for round={round_id}, remaining={remaining}"
        )

    final_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    if final_tokens != target_tokens:
        raise RuntimeError(
            f"final token count mismatch: got={final_tokens}, target={target_tokens}"
        )
    return text


def build_text(
    tokenizer: AutoTokenizer,
    round_id: int,
    target_tokens: int,
) -> tuple[str, int]:
    random.seed(1000 + round_id)
    text = ""
    sentence_id = 0
    while True:
        candidate = text + build_sentence(round_id, sentence_id)
        candidate_tokens = len(tokenizer.encode(candidate, add_special_tokens=False))
        if candidate_tokens > target_tokens:
            text = fit_tail_exact(
                tokenizer,
                text,
                round_id,
                sentence_id,
                target_tokens,
            )
            break
        text = candidate
        sentence_id += 1

    token_count = len(tokenizer.encode(text, add_special_tokens=False))
    if token_count != target_tokens:
        raise RuntimeError(
            f"unexpected token count for round {round_id}: {token_count}"
        )
    return text, token_count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_text_uniqueness(text: str, round_id: int) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != len(set(lines)):
        raise RuntimeError(f"round_{round_id} generated duplicate text lines")
    return lines


def generate_unique_image(
    path: Path,
    size: tuple[int, int],
    round_id: int,
    image_id: int,
    seed: int,
) -> None:
    width, height = size
    image = Image.new("RGB", size)
    pixels = image.load()

    global_index = round_id * 1000 + image_id
    seed_mix = seed + round_id * 7919 + image_id * 1543

    for y in range(height):
        row_term = (seed_mix + y * (round_id + 11)) % 256
        for x in range(width):
            pixels[x, y] = (
                (x * 5 + y * 3 + row_term + global_index) % 256,
                (x * 7 + y * 11 + seed_mix // 7 + image_id * 17) % 256,
                ((x ^ y) + seed_mix // 13 + round_id * 19 + image_id * 29) % 256,
            )

    draw = ImageDraw.Draw(image)
    border_color = (
        (global_index * 53) % 256,
        (global_index * 97) % 256,
        (global_index * 149) % 256,
    )
    draw.rectangle((0, 0, width - 1, height - 1), outline=border_color, width=3)

    step = max(width // 6, 24)
    for column in range(0, width, step):
        line_color = (
            (column + round_id * 31) % 256,
            (column * 3 + image_id * 47) % 256,
            (column * 5 + seed_mix) % 256,
        )
        draw.line(
            (column, 0, width - 1 - (column % width), height - 1),
            fill=line_color,
            width=2,
        )

    draw.text((12, 12 + (round_id % 5) * 18), f"R{round_id:02d}-I{image_id:02d}", fill=(255, 255, 255))

    marker_bytes = global_index.to_bytes(4, byteorder="little", signed=False)
    for offset, value in enumerate(marker_bytes):
        pixels[offset, 0] = (
            value,
            (value * 53 + round_id) % 256,
            (value * 97 + image_id) % 256,
        )
    pixels[4, 0] = (round_id, image_id, (round_id + image_id) % 256)

    image.save(path, "PNG")


def build_payload(
    round_dir: Path,
    text: str,
    request_model: str,
    images_per_round: int,
    max_completion_tokens: int,
) -> dict:
    image_paths = sorted((round_dir / "images").glob("*.png"))
    if len(image_paths) != images_per_round:
        raise RuntimeError(
            f"{round_dir} expected {images_per_round} images, found {len(image_paths)}"
        )

    content = [{"type": "text", "text": text}]
    for image_path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_path.resolve().as_uri()},
            }
        )

    return {
        "model": request_model,
        "messages": [{"role": "user", "content": content}],
        "max_completion_tokens": max_completion_tokens,
        "temperature": 0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    seen_texts: set[str] = set()
    seen_image_hashes: dict[str, str] = {}
    round_summaries: list[dict[str, object]] = []

    for round_id in range(args.rounds):
        round_dir = output_dir / f"round_{round_id}"
        image_dir = round_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        text, token_count = build_text(tokenizer, round_id, args.target_text_tokens)
        text_lines = validate_text_uniqueness(text, round_id)
        if text in seen_texts:
            raise RuntimeError(f"round_{round_id} duplicated a previous round text")
        seen_texts.add(text)

        image_paths: list[Path] = []
        image_hashes: list[str] = []
        for image_id in range(args.images_per_round):
            image_path = image_dir / f"img_{image_id:02d}.png"
            generate_unique_image(
                image_path,
                (args.width, args.height),
                round_id,
                image_id,
                args.seed,
            )
            image_hash = sha256_file(image_path)
            if image_hash in seen_image_hashes:
                raise RuntimeError(
                    "duplicate image bytes detected between "
                    f"{seen_image_hashes[image_hash]} and {image_path.resolve()}"
                )
            seen_image_hashes[image_hash] = str(image_path.resolve())
            image_paths.append(image_path.resolve())
            image_hashes.append(image_hash)

        payload = build_payload(
            round_dir,
            text,
            args.request_model,
            args.images_per_round,
            args.max_completion_tokens,
        )
        payload_path = round_dir / "payload.json"
        payload_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        round_summaries.append(
            {
                "round": round_id,
                "text_tokens": token_count,
                "text_line_count": len(text_lines),
                "image_count": len(image_paths),
                "payload_path": str(payload_path.resolve()),
                "image_sha256": image_hashes,
            }
        )
        print(
            f"round_{round_id}: text_tokens={token_count} "
            f"images={len(image_paths)} payload={payload_path.resolve()}"
        )

    manifest = {
        "tokenizer": args.tokenizer,
        "request_model": args.request_model,
        "output_dir": str(output_dir),
        "rounds": args.rounds,
        "warmup_rounds": list(range(args.warmup_rounds)),
        "test_rounds": list(range(args.warmup_rounds, args.rounds)),
        "target_text_tokens": args.target_text_tokens,
        "images_per_round": args.images_per_round,
        "image_size": {"width": args.width, "height": args.height},
        "max_completion_tokens": args.max_completion_tokens,
        "seed": args.seed,
        "unique_text_count": len(seen_texts),
        "unique_image_count": len(seen_image_hashes),
        "round_summaries": round_summaries,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"manifest={manifest_path.resolve()}")


if __name__ == "__main__":
    main()
