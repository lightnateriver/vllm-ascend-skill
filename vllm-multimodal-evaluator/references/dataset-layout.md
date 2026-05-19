# Dataset Layout

Use this reference when creating or modifying the evaluator fixtures.

## Image layout

```text
pics/<resolution>/<format>/<shape>.<ext>
```

Typical examples:

```text
pics/720x1280/jpg/rectangle.jpg
pics/4096x6144/jpg/circle.jpg
pics/4096x8192/png/triangle.png
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

- blue shape on green background

Default image formats:

- `jpg`
- `png`
- `webp`
- `bmp`
- `tiff`

## Video layout

```text
video/<resolution>/<format>/shapes.<ext>
```

Multi-video understanding clips are stored as:

```text
video/720x1280/mp4/<shape>.mp4
```

Typical examples:

```text
video/720x1280/mp4/shapes.mp4
video/720x1280/mp4/square.mp4
video/1080x1920/mkv/shapes.mkv
video/4096x6144/mp4/shapes.mp4
```

Default video properties:

- source frames come from the standard image set
- one second per shape
- `16 fps`
- low-size encoding
- `crf=32`

## Resolution profiles

### Image

- `standard`
  - `256x512`
  - `720x1280`
  - `1920x1080`
- `large`
  - `4096x4096`
  - `4096x6144`
  - `4096x8192`

### Video

- `standard`
  - `720x1280`
  - `1080x1920`
- `large`
  - `4096x4096`
  - `4096x6144`
  - `4096x8192`

Large videos default to `mp4` only to keep generation practical.
Single-shape clips for multi-video understanding default to `720x1280/mp4` only.

## Naming rules

- keep English shape names stable
- keep shape order stable:
  `square, rectangle, rhombus, circle, triangle, cylinder, cube`
- if a resolution is removed, update the generator and report references together
