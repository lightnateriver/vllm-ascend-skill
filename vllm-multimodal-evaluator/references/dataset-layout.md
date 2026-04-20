# Dataset Layout

Use this reference when creating or modifying the synthetic evaluation fixtures.

## Image dataset

Default image directory layout:

```text
pics/<resolution>/<format>/<shape>.<ext>
```

Example:

```text
pics/720x1280/jpg/rectangle.jpg
pics/1920x1080/tiff/circle.tiff
```

Default shapes:

- `square`
- `rectangle`
- `rhombus`
- `circle`
- `triangle`
- `cylinder`
- `cube`

Default colors:

- shape fill: blue
- background: green

Default image formats:

- `jpg`
- `png`
- `webp`
- `bmp`
- `tiff`

Common tested resolutions in this skill:

- `256x512`
- `720x1280`
- `1920x1080`

## Video dataset

Default video directory layout:

```text
video/<resolution>/<format>/shapes.<ext>
```

Example:

```text
video/720x1280/mp4/shapes.mp4
video/1080x1920/mkv/shapes.mkv
```

Default video properties:

- source frames come from `pics/720x1280/jpg/*.jpg`
- one second per shape
- `16 fps`
- low-size encoding
- `crf=32`

Default video formats:

- `mp4`
- `avi`
- `mov`
- `mkv`

Default target resolutions:

- `720x1280`
- `1080x1920`

## Naming rules

- Keep English shape names stable because the evaluation script depends on them.
- Keep the shape order stable:
  `square, rectangle, rhombus, circle, triangle, cylinder, cube`
- If a user deletes a large resolution to save space, update the generator config rather than leaving stale references in the workflow.
