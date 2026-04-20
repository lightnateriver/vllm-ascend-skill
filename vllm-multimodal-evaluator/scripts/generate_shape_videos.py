from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
from PIL import Image


SOURCE_RESOLUTION = "720x1280"
TARGET_RESOLUTIONS = {
    "720x1280": (720, 1280),
    "1080x1920": (1080, 1920),
}
VIDEO_FORMATS = ["mp4", "avi", "mov", "mkv"]
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


def main() -> None:
    project_root = Path.cwd()
    source_dir = project_root / "pics" / SOURCE_RESOLUTION / "jpg"
    output_root = project_root / "video"
    images = load_source_images(source_dir)

    generated_count = 0
    for resolution_name, (width, height) in TARGET_RESOLUTIONS.items():
        for extension in VIDEO_FORMATS:
            output_path = output_root / resolution_name / extension / f"shapes.{extension}"
            encode_video(images, output_path, width, height)
            generated_count += 1

    print(f"Generated {generated_count} videos under {output_root}")


if __name__ == "__main__":
    main()
