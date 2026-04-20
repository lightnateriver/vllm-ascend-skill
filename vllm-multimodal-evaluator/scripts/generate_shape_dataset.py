from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


BACKGROUND_COLOR = (0, 255, 0)
SHAPE_FILL_COLOR = (0, 0, 255)
SHAPE_OUTLINE_COLOR = (0, 0, 180)

RESOLUTIONS = [
    (1920, 1080),
    (720, 1280),
    (256, 512),
]

FORMATS = {
    "jpg": "JPEG",
    "png": "PNG",
    "webp": "WEBP",
    "bmp": "BMP",
    "tiff": "TIFF",
}

SHAPES = [
    "square",
    "rectangle",
    "rhombus",
    "circle",
    "triangle",
    "cylinder",
    "cube",
]


def draw_square(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    side = int(min(width, height) * 0.5)
    left = (width - side) // 2
    top = (height - side) // 2
    draw.rectangle([left, top, left + side, top + side], fill=SHAPE_FILL_COLOR)


def draw_rectangle(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    rect_w = int(width * 0.56)
    rect_h = int(height * 0.42)
    left = (width - rect_w) // 2
    top = (height - rect_h) // 2
    draw.rectangle([left, top, left + rect_w, top + rect_h], fill=SHAPE_FILL_COLOR)


def draw_rhombus(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    rhombus_w = int(width * 0.55)
    rhombus_h = int(height * 0.5)
    cx = width // 2
    cy = height // 2
    points = [
        (cx, cy - rhombus_h // 2),
        (cx + rhombus_w // 2, cy),
        (cx, cy + rhombus_h // 2),
        (cx - rhombus_w // 2, cy),
    ]
    draw.polygon(points, fill=SHAPE_FILL_COLOR)


def draw_circle(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    diameter = int(min(width, height) * 0.52)
    left = (width - diameter) // 2
    top = (height - diameter) // 2
    draw.ellipse([left, top, left + diameter, top + diameter], fill=SHAPE_FILL_COLOR)


def draw_triangle(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    tri_w = int(width * 0.56)
    tri_h = int(height * 0.56)
    cx = width // 2
    cy = height // 2
    points = [
        (cx, cy - tri_h // 2),
        (cx + tri_w // 2, cy + tri_h // 2),
        (cx - tri_w // 2, cy + tri_h // 2),
    ]
    draw.polygon(points, fill=SHAPE_FILL_COLOR)


def draw_cylinder(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    body_w = int(width * 0.42)
    body_h = int(height * 0.46)
    ellipse_h = max(24, int(height * 0.12))
    left = (width - body_w) // 2
    top = (height - body_h) // 2
    right = left + body_w
    bottom = top + body_h
    line_width = max(2, min(width, height) // 300)

    draw.rectangle(
        [left, top + ellipse_h // 2, right, bottom - ellipse_h // 2],
        fill=SHAPE_FILL_COLOR,
        outline=SHAPE_OUTLINE_COLOR,
        width=line_width,
    )
    draw.ellipse(
        [left, top, right, top + ellipse_h],
        fill=SHAPE_FILL_COLOR,
        outline=SHAPE_OUTLINE_COLOR,
        width=line_width,
    )
    draw.ellipse(
        [left, bottom - ellipse_h, right, bottom],
        fill=SHAPE_FILL_COLOR,
        outline=SHAPE_OUTLINE_COLOR,
        width=line_width,
    )


def draw_cube(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    face = int(min(width, height) * 0.34)
    offset = int(face * 0.28)
    line_width = max(2, min(width, height) // 300)

    front_left = (width - face) // 2 - offset // 2
    front_top = (height - face) // 2 + offset // 3
    back_left = front_left + offset
    back_top = front_top - offset

    front = [
        (front_left, front_top),
        (front_left + face, front_top),
        (front_left + face, front_top + face),
        (front_left, front_top + face),
    ]
    back = [
        (back_left, back_top),
        (back_left + face, back_top),
        (back_left + face, back_top + face),
        (back_left, back_top + face),
    ]

    top_face = [back[0], back[1], front[1], front[0]]
    side_face = [front[1], back[1], back[2], front[2]]

    draw.polygon(back, fill=SHAPE_FILL_COLOR, outline=SHAPE_OUTLINE_COLOR, width=line_width)
    draw.polygon(top_face, fill=SHAPE_FILL_COLOR, outline=SHAPE_OUTLINE_COLOR, width=line_width)
    draw.polygon(side_face, fill=SHAPE_FILL_COLOR, outline=SHAPE_OUTLINE_COLOR, width=line_width)
    draw.polygon(front, fill=SHAPE_FILL_COLOR, outline=SHAPE_OUTLINE_COLOR, width=line_width)


DRAWERS = {
    "square": draw_square,
    "rectangle": draw_rectangle,
    "rhombus": draw_rhombus,
    "circle": draw_circle,
    "triangle": draw_triangle,
    "cylinder": draw_cylinder,
    "cube": draw_cube,
}


def save_image(image: Image.Image, output_path: Path, extension: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, object] = {"format": FORMATS[extension]}
    if extension in {"jpg", "webp"}:
        save_kwargs["quality"] = 95
    image.save(output_path, **save_kwargs)


def render_shape(shape_name: str, width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)
    DRAWERS[shape_name](draw, width, height)
    return image


def main() -> None:
    project_root = Path.cwd()
    output_root = project_root / "pics"

    generated_count = 0
    for width, height in RESOLUTIONS:
        resolution_dir = output_root / f"{width}x{height}"
        for extension in FORMATS:
            format_dir = resolution_dir / extension
            for shape_name in SHAPES:
                image = render_shape(shape_name, width, height)
                output_path = format_dir / f"{shape_name}.{extension}"
                save_image(image, output_path, extension)
                generated_count += 1

    print(f"Generated {generated_count} images under {output_root}")


if __name__ == "__main__":
    main()
