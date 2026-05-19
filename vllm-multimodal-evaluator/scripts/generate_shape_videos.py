from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import av
from PIL import Image


SOURCE_RESOLUTION = "720x1280"
STANDARD_TARGET_RESOLUTIONS = {
    "720x1280": (720, 1280),
    "1080x1920": (1080, 1920),
}
LARGE_TARGET_RESOLUTIONS = {
    "4096x4096": (4096, 4096),
    "4096x6144": (4096, 6144),
    "4096x8192": (4096, 8192),
}
STANDARD_VIDEO_FORMATS = ["mp4", "avi", "mov", "mkv"]
LARGE_VIDEO_FORMATS = ["mp4"]
SHAPES = [
    "square",
    "rectangle",
    "rhombus",
    "circle",
    "triangle",
    "cylinder",
    "cube",
]

FPS = 16
SECONDS_PER_SHAPE = 1
CRF = "32"


def parse_resolution(value: str) -> tuple[str, tuple[int, int]]:
    width_str, height_str = value.lower().split("x", 1)
    width = int(width_str)
    height = int(height_str)
    if width <= 0 or height <= 0:
        raise ValueError(f"Resolution must be positive: {value}")
    normalized = f"{width}x{height}"
    return normalized, (width, height)


def unique_resolution_map(items: list[tuple[str, tuple[int, int]]]) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for key, value in items:
        result[key] = value
    return result


def load_source_images(source_dir: Path) -> list[Image.Image]:
    images: list[Image.Image] = []
    missing_paths: list[Path] = []

    for shape in SHAPES:
        image_path = source_dir / f"{shape}.jpg"
        if not image_path.exists():
            missing_paths.append(image_path)
            continue
        images.append(Image.open(image_path).convert("RGB"))

    if missing_paths:
        missing = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing source image(s):\n{missing}")

    return images


def resize_image(image: Image.Image, width: int, height: int) -> Image.Image:
    if image.size == (width, height):
        return image
    return image.resize((width, height), Image.Resampling.BICUBIC)


def encode_video(images: list[Image.Image], output_path: Path, width: int, height: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with av.open(output_path, mode="w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, FPS)
        stream.options = {
            "crf": CRF,
            "preset": "veryslow",
            "tune": "stillimage",
        }

        for source_image in images:
            resized_image = resize_image(source_image, width, height)
            for _ in range(FPS * SECONDS_PER_SHAPE):
                frame = av.VideoFrame.from_image(resized_image)
                for packet in stream.encode(frame):
                    container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)


def resolve_resolution_map(profile: str, explicit: list[str]) -> dict[str, tuple[int, int]]:
    items: list[tuple[str, tuple[int, int]]] = []
    if profile == "standard":
        items.extend(STANDARD_TARGET_RESOLUTIONS.items())
    elif profile == "large":
        items.extend(LARGE_TARGET_RESOLUTIONS.items())
    else:
        items.extend(STANDARD_TARGET_RESOLUTIONS.items())
        items.extend(LARGE_TARGET_RESOLUTIONS.items())
    items.extend(parse_resolution(item) for item in explicit)
    return unique_resolution_map(items)


def resolve_formats(profile: str, explicit: list[str]) -> list[str]:
    if explicit:
        formats = [item.lower() for item in explicit]
    elif profile == "large":
        formats = list(LARGE_VIDEO_FORMATS)
    elif profile == "standard":
        formats = list(STANDARD_VIDEO_FORMATS)
    else:
        formats = list(dict.fromkeys(STANDARD_VIDEO_FORMATS + LARGE_VIDEO_FORMATS))
    unsupported = [item for item in formats if item not in STANDARD_VIDEO_FORMATS]
    if unsupported:
        raise ValueError(f"Unsupported video format(s): {', '.join(unsupported)}")
    return formats


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate synthetic shape videos for evaluator capability tests.")
    parser.add_argument(
        "--resolution-profile",
        choices=("standard", "large", "all"),
        default="all",
        help="Which built-in resolution set to generate.",
    )
    parser.add_argument(
        "--resolutions",
        nargs="*",
        default=[],
        help="Optional extra resolutions in WIDTHxHEIGHT format.",
    )
    parser.add_argument(
        "--formats",
        nargs="*",
        default=[],
        help="Optional subset of video container formats to generate.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=project_root / "pics",
        help="Input image root. The script reads JPG frames from <source-root>/720x1280/jpg.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "video",
        help="Output root for generated videos.",
    )
    args = parser.parse_args()

    source_dir = args.source_root.resolve() / SOURCE_RESOLUTION / "jpg"
    output_root = args.output_root.resolve()
    images = load_source_images(source_dir)
    resolutions = resolve_resolution_map(args.resolution_profile, args.resolutions)
    formats = resolve_formats(args.resolution_profile, args.formats)

    generated_count = 0
    for resolution_name, (width, height) in resolutions.items():
        for extension in formats:
            output_path = output_root / resolution_name / extension / f"shapes.{extension}"
            encode_video(images, output_path, width, height)
            generated_count += 1

    print(f"Generated {generated_count} videos under {output_root}")


if __name__ == "__main__":
    main()
