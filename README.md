<img width="1638" height="959" alt="Screenshot 2026-03-16 at 23 14 31" src="https://github.com/user-attachments/assets/c0aba84c-3836-4b6a-a3b6-a7b9a853b592" />



# TeslaCamOverlay

Python tool that reads Tesla dashcam MP4 files, extracts embedded SEI telemetry
frame-by-frame, and produces a new MP4 with:

* the original video preserved on the left
* a telemetry data table rendered in a dedicated right-side panel
* a compact glass-style HUD overlay drawn directly on the video

Telemetry extraction is a direct port of Tesla's `DashcamMP4.parseFrames()`
logic from their public repository, which walks the MP4 `mdat` atom and
atomically binds each SEI protobuf message to its corresponding video frame.
Video decoding uses [PyAV](https://pyav.org/) for per-frame decode control
(mirroring Tesla's WebCodecs `VideoDecoder` approach), while encoding uses
ffmpeg for the output MP4.

Source: <https://github.com/teslamotors/dashcam>

## Requirements

* Python 3.10+
* `ffmpeg` available on `PATH` (used only for encoding the output)
* Python packages from `requirements.txt`:
  * `numpy`, `opencv-python-headless`, `Pillow`, `protobuf`, `av` (PyAV)

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python3 tesla_dashcam_overlay.py input.mp4 output.mp4
```

### Common flags

```bash
python3 tesla_dashcam_overlay.py input.mp4 output.mp4 \
  --panel-width 420 \
  --speed-unit kph \
  --hud-position top-right \
  --hud-transparency 0.54 \
  --telemetry-offset-seconds 0.30 \
  --crf 4 \
  --quality-mode visually-lossless
```

### Image enhancement

Enable with the bundled default profile:

```bash
python3 tesla_dashcam_overlay.py input.mp4 output.mp4 \
  --enhance-video \
  --opencv-accel auto
```

Use a custom JSON profile:

```bash
python3 tesla_dashcam_overlay.py input.mp4 output.mp4 \
  --image-params-json /path/to/other_image_params.json
```

Inline JSON also works:

```bash
python3 tesla_dashcam_overlay.py input.mp4 output.mp4 \
  --image-params-json '{"gamma":1.35,"contrast":1.12}'
```

## HUD

The in-frame HUD is a compact glass card with warm dark tones, rendered in a
corner of the video. It displays:

* **Gear** — color-coded pill (blue=Drive, coral=Reverse, gold=Neutral)
* **Steering** — wheel icon with rotation line + angle in degrees
* **Speed** — large hero number with unit label
* **Blinkers** — left/right arrow indicators in rounded pills (amber when active)
* **Brake** — pill indicator with glow when pressed
* **Accelerator** — progress bar (teal < 80%, amber above) with percentage
* **G-force** — circle with crosshair and moving dot + numeric value

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--panel-width` | `360` | Width of the right-side telemetry panel in pixels |
| `--speed-unit` | `mph` | Speed display unit: `mph`, `kph`, or `mps` |
| `--hud-position` | `top-right` | HUD corner: `top-left`, `top-right`, `bottom-left`, `bottom-right` |
| `--hud-transparency` | `0.54` | HUD background opacity (`0.0` opaque to `1.0` transparent) |
| `--telemetry-offset-frames` | `0` | Shift telemetry by whole frames (positive = earlier) |
| `--telemetry-offset-seconds` | `0.0` | Shift telemetry by seconds (positive = earlier) |
| `--crf` | `4` | libx264 CRF value (lower = higher quality, larger files) |
| `--quality-mode` | `visually-lossless` | `visually-lossless` (4:4:4), `lossless` (CRF 0), or `compatible` (yuv420p) |
| `--gpu` | `auto` | Hardware encoder selection (macOS/Windows, `compatible` mode only) |
| `--enhance-video` | off | Enable image enhancement with the bundled default profile |
| `--image-params-json` | — | Path or inline JSON for a custom image enhancement profile |
| `--opencv-accel` | `auto` | OpenCV acceleration: `auto`, `on`, or `off` |

## Notes

* Tesla SEI data is only present on supported clips. Per Tesla's README, that
  generally means firmware `2025.44.25+`, HW3+, and non-parked clips.
* PyAV is the sole video decoder. ffmpeg is only used for encoding the output.
* The right-side data table does not cover the video. The in-frame HUD only
  adds compact visual indicators on a semi-transparent card.
* `--quality-mode visually-lossless` is the default and keeps 4:4:4 chroma
  to avoid the softening you get from `yuv420p`.
* Use `--quality-mode lossless` to avoid any encoder loss at the cost of very
  large files. Use `--quality-mode compatible` only if you need broad player
  compatibility.
* The bundled default image profile is stored in `default_image_params.json`
  and is used when you pass `--enhance-video`.
* `--image-params-json` accepts either a file path or an inline JSON object.
  When provided, it enables enhancement and replaces the default profile.
* `--opencv-accel auto` uses OpenCL/`UMat` when available, otherwise CPU.
  Only matters when image enhancement is enabled.
* `--gpu auto` picks a hardware encoder on macOS/Windows but only for
  `--quality-mode compatible`. Higher-quality modes stay on CPU to preserve
  image fidelity.
