#!/usr/bin/env python3
"""Generate a deterministic 1-to-40 multi-image precision dataset."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw


SEED = 20260426
WIDTH = 1280
HEIGHT = 720
BACKGROUND = (245, 245, 245)
OUTLINE = (30, 30, 30)


ColorMap = dict[str, tuple[int, int, int]]

COLORS: ColorMap = {
    "red": (220, 45, 45),
    "blue": (35, 95, 220),
    "green": (35, 160, 85),
    "yellow": (235, 200, 35),
    "orange": (235, 130, 35),
    "purple": (140, 70, 200),
    "black": (40, 40, 40),
    "white": (250, 250, 250),
    "brown": (145, 95, 55),
    "pink": (230, 120, 170),
}

SHAPES = [
    "rectangle",
    "square",
    "triangle",
    "circle",
    "rhombus",
    "cube",
    "cylinder",
    "cone",
]


@dataclass(frozen=True)
class ShapeItem:
    index: int
    shape: str
    color: str

    @property
    def stem(self) -> str:
        return f"{self.index:02d}_{self.color}_{self.shape}"


def clamp_channel(value: int) -> int:
    return max(0, min(255, value))


def adjust_color(rgb: tuple[int, int, int], delta: int) -> tuple[int, int, int]:
    return tuple(clamp_channel(channel + delta) for channel in rgb)


def bounds_for_shape() -> tuple[int, int, int, int]:
    return (320, 140, 960, 580)


def draw_rectangle(draw: ImageDraw.ImageDraw, fill: tuple[int, int, int]) -> None:
    draw.rounded_rectangle((360, 220, 920, 500), radius=18, fill=fill, outline=OUTLINE, width=8)


def draw_square(draw: ImageDraw.ImageDraw, fill: tuple[int, int, int]) -> None:
    draw.rounded_rectangle((420, 180, 860, 620), radius=18, fill=fill, outline=OUTLINE, width=8)


def draw_triangle(draw: ImageDraw.ImageDraw, fill: tuple[int, int, int]) -> None:
    points = [(640, 150), (930, 560), (350, 560)]
    draw.polygon(points, fill=fill, outline=OUTLINE)
    draw.line(points + [points[0]], fill=OUTLINE, width=8)


def draw_circle(draw: ImageDraw.ImageDraw, fill: tuple[int, int, int]) -> None:
    draw.ellipse((360, 140, 920, 700), fill=fill, outline=OUTLINE, width=8)


def draw_rhombus(draw: ImageDraw.ImageDraw, fill: tuple[int, int, int]) -> None:
    points = [(640, 140), (950, 360), (640, 580), (330, 360)]
    draw.polygon(points, fill=fill, outline=OUTLINE)
    draw.line(points + [points[0]], fill=OUTLINE, width=8)


def draw_cube(draw: ImageDraw.ImageDraw, fill: tuple[int, int, int]) -> None:
    front = [(420, 240), (760, 240), (760, 560), (420, 560)]
    offset_x = 140
    offset_y = -90
    back = [(x + offset_x, y + offset_y) for x, y in front]
    top = [back[0], back[1], front[1], front[0]]
    side = [front[1], back[1], back[2], front[2]]
    draw.polygon(top, fill=adjust_color(fill, 35), outline=OUTLINE)
    draw.polygon(side, fill=adjust_color(fill, -25), outline=OUTLINE)
    draw.polygon(front, fill=fill, outline=OUTLINE)
    for start, end in zip(front, back):
        draw.line([start, end], fill=OUTLINE, width=8)


def draw_cylinder(draw: ImageDraw.ImageDraw, fill: tuple[int, int, int]) -> None:
    top_box = (400, 170, 880, 290)
    bottom_box = (400, 430, 880, 550)
    body_fill = adjust_color(fill, -10)
    draw.rectangle((400, 230, 880, 490), fill=body_fill, outline=OUTLINE, width=8)
    draw.ellipse(top_box, fill=adjust_color(fill, 25), outline=OUTLINE, width=8)
    draw.ellipse(bottom_box, fill=fill, outline=OUTLINE, width=8)
    draw.line([(400, 230), (400, 490)], fill=OUTLINE, width=8)
    draw.line([(880, 230), (880, 490)], fill=OUTLINE, width=8)


def draw_cone(draw: ImageDraw.ImageDraw, fill: tuple[int, int, int]) -> None:
    apex = (640, 150)
    left = (390, 530)
    right = (890, 530)
    draw.polygon([apex, right, left], fill=fill, outline=OUTLINE)
    draw.line([apex, left], fill=OUTLINE, width=8)
    draw.line([apex, right], fill=OUTLINE, width=8)
    draw.ellipse((390, 470, 890, 590), fill=adjust_color(fill, -20), outline=OUTLINE, width=8)


DRAWERS: dict[str, Callable[[ImageDraw.ImageDraw, tuple[int, int, int]], None]] = {
    "rectangle": draw_rectangle,
    "square": draw_square,
    "triangle": draw_triangle,
    "circle": draw_circle,
    "rhombus": draw_rhombus,
    "cube": draw_cube,
    "cylinder": draw_cylinder,
    "cone": draw_cone,
}


def make_image(item: ShapeItem, output_path: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    fill = COLORS[item.color]
    drawer = DRAWERS[item.shape]
    drawer(draw, fill)
    image.save(output_path, format="JPEG", quality=95)


def build_question(case_id: int, items: list[ShapeItem], target: ShapeItem) -> tuple[str, str]:
    if case_id == 1:
        question = (
            f"Is this a {target.color} {target.shape}? "
            "Answer only one word: YES or NO."
        )
        answer = "YES"
    else:
        question = (
            f"Which image is the {target.color} {target.shape}? "
            "Answer only one number."
        )
        answer = str(target.index)
    return question, answer


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def write_case(case_dir: Path, case_id: int, items: list[ShapeItem], target: ShapeItem) -> dict:
    question, answer = build_question(case_id, items, target)
    for item in items:
        make_image(item, case_dir / f"{item.stem}.jpg")

    question_md = "\n".join(
        [
            f"# Case {case_id:02d}",
            "",
            question,
            "",
            "Output rule:",
            "- Reply with exactly the requested short answer.",
        ]
    )
    write_text(case_dir / "question.md", question_md + "\n")

    answer_lines = [
        f"# Case {case_id:02d} Answer",
        "",
        f"- Standard answer: `{answer}`",
        f"- Target shape: `{target.shape}`",
        f"- Target color: `{target.color}`",
        f"- Target image index: `{target.index}`",
        "",
        "## Image Inventory",
    ]
    answer_lines.extend(
        f"- {item.index:02d}: `{item.color} {item.shape}`" for item in items
    )
    write_text(case_dir / "answer.md", "\n".join(answer_lines) + "\n")

    metadata = {
        "case_id": case_id,
        "image_count": len(items),
        "question_type": "yes_no" if case_id == 1 else "index",
        "question": question,
        "answer": answer,
        "target": {
            "index": target.index,
            "shape": target.shape,
            "color": target.color,
            "filename": f"{target.stem}.jpg",
        },
        "images": [
            {
                "index": item.index,
                "shape": item.shape,
                "color": item.color,
                "filename": f"{item.stem}.jpg",
            }
            for item in items
        ],
    }
    write_text(
        case_dir / "answer.json",
        json.dumps(metadata, ensure_ascii=True, indent=2) + "\n",
    )
    return metadata


def build_dataset(dataset_dir: Path) -> None:
    rng = random.Random(SEED)
    all_pairs = list(product(SHAPES, COLORS.keys()))

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True)

    readme = "\n".join(
        [
            "# Multi Pics Dataset",
            "",
            "Deterministic multi-image precision dataset for 1 to 40 image requests.",
            "",
            "Rules:",
            "- Case `01` contains 1 image and uses a `YES` or `NO` question.",
            "- Cases `02` to `40` contain N images and ask for exactly one target image index.",
            "- Shape and color combinations are unique within each case.",
            "- A shape may repeat inside a case, but repeated shapes always use different colors.",
            "- Files are generated by `generate_dataset.py` with seed `20260426`.",
            "",
            "Included files in each case directory:",
            "- `question.md`",
            "- `answer.md`",
            "- `answer.json`",
            "- `*.jpg` images ordered by filename prefix",
        ]
    )
    write_text(dataset_dir / "README.md", readme + "\n")

    manifests = []
    for case_id in range(1, 41):
        case_dir = dataset_dir / f"{case_id:02d}"
        case_dir.mkdir()
        sample_pairs = rng.sample(all_pairs, k=case_id)
        items = [
            ShapeItem(index=index, shape=shape, color=color)
            for index, (shape, color) in enumerate(sample_pairs, start=1)
        ]
        target = rng.choice(items)
        manifests.append(write_case(case_dir, case_id, items, target))

    summary = {
        "seed": SEED,
        "width": WIDTH,
        "height": HEIGHT,
        "colors": list(COLORS.keys()),
        "shapes": SHAPES,
        "case_count": 40,
        "cases": manifests,
    }
    write_text(dataset_dir / "manifest.json", json.dumps(summary, indent=2) + "\n")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    cases_dir = base_dir / "cases"
    build_dataset(cases_dir)
    print(f"Generated dataset under: {cases_dir}")


if __name__ == "__main__":
    main()
