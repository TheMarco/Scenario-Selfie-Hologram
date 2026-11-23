# HoloMe CLI

HoloMe renders talking 3D hologram videos of a person. It rigs a 3D bust of them (from a Tripo `.glb` or a three-view character sheet) and drives it with a selfie video, an audio file, or a generated talking-head clip. Each render can write Apple ProRes 4444 with alpha (`.mov`) and a black-background H.264 (`.mp4`) for Screen or Add blending in one pass.

This is the tool behind the Scenario Selfie Hologram skill. The skill (`../SKILL.md`) runs the Scenario steps; everything below runs locally.

## Setup

```bash
./setup.sh          # Python 3.12 venv, Node packages, MediaPipe model, renderer bundle, model weights
./holome doctor     # checks every dependency and says what is missing
```

Needs node 18+ with npm, ffmpeg with ffprobe, curl, uv or python3.12, and Google Chrome, Chromium or Brave (setup installs Playwright Chromium if none is found). Tested on macOS with Apple Silicon.

## Workflow for one person

```bash
P=~/HoloMe/jane-doe

# 1. Reference still from a selfie video (or use photos directly)
./holome pick-frame --video $P/videos/take1.MOV --out $P/photos/video_frame.jpg

# 2. Generate $P/sheet.png on Scenario (GPT Image 2.5 Sunburst, with assets/templates/sheet_template.png
#    as the layout reference), then split it and check qa_panels.png
./holome split-sheet --sheet $P/sheet.png --out-dir $P/panels

# 3. Generate $P/head.glb on Scenario (Tripo P2 Multi View from the three panel_*_hires.png, faceLimit 50000),
#    then rig it
./holome import-glb --glb $P/head.glb --out $P/bust

# 4. Check a still, then render a selfie video as a hologram
./holome render --bust $P/bust --out $P/previews/hero.png --preview-frame 0
./holome render --bust $P/bust --mocap-video $P/videos/take1.MOV --out $P/renders/take1
```

Every command accepts `--json`: one JSON object goes to stdout and progress goes to stderr.

## `render`

Always pass `--bust`. Pick one input:

| Input | What drives the hologram |
|---|---|
| `-m, --mocap-video <path>` | A talking-head video of the person (iPhone `.MOV` works). Every frame is tracked (52 blendshapes plus head pose), lips are timed to the speech with Whisper phonemes, and the video's audio becomes the voice. Recommended. |
| `-a, --audio <path>` | Spoken audio only (`.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`). Lips follow the speech; blinks and head motion are procedural. With `--mocap-video` or `--driving-video` it replaces that video's audio. |
| `-d, --driving-video <path>` | A talking-head video whose face pixels are projected onto the bust. |
| `--preview-frame <sec>` | A single PNG instead of a video. Without audio it shows a neutral pose; `--preview-jaw`, `--preview-funnel`, `--preview-wide` and `--preview-blink` (0 to 1) pose the face. |

| Flag | Default | Description |
|---|---|---|
| `--bust <dir>` | `assets/bust` | Bust folder from `import-glb` or `build-bust`, one per person |
| `-o, --out <path>` | required | Output file or stem; `--format both` writes `<stem>.mov` and `<stem>.mp4` |
| `-f, --format <type>` | `both` | `both`, `mov` (ProRes 4444 alpha), `mp4`, `webm` (VP9 alpha) or `png` |
| `-r, --resolution <res>` | `1080p` | `720p`, `1080p`, `4k`, `vertical` (1080 x 1920 with a 9:16 HUD layout), `square` or `WxH` |
| `--fps <num>` | `30` | `24`, `30` or `60` |
| `--text <string>` | none | Transcript hint for phoneme alignment (helps with names) |
| `--transcript <path>` | none | Transcript from a text file |
| `--camera <preset>` | `hero` | `hero` (3/4), `front`, `left34`, `right34`, `orbit` |
| `--camera-motion <mode>` | `subtle` | `subtle`, `static`, `orbit` |
| `--camera-yaw <deg>`, `--camera-pitch <deg>` | none | Custom camera angle |
| `--camera-zoom <num>` | `1.0` | `1.15` closer, `0.85` wider |
| `--no-look-at-camera` | off | Keep the head facing forward instead of toward the camera |
| `--no-hud` | off | Hide the floating HUD graphics |
| `--no-emitter` | off | Hide the projector base and light cone (clean floating bust for overlays) |
| `--background <mode>` | `auto` | `auto` (transparent for `both`, `mov`, `webm`; black otherwise), `transparent`, `black`, `lab` |
| `--color <spec>` | `cyan` | `cyan`, `ice`, `amber`, `emerald`, `crimson`, `violet` or `#RRGGBB` |
| `--bloom <num>` | `1.0` | Glow strength, 0 to 2 |
| `--glitch <num>` | `0.35` | Audio-reactive glitch, 0 to 1 |
| `--density <num>` | `1.0` | Point density, 0.3 to 1 |
| `--intensity <num>` | `1.0` | Overall brightness |
| `--point-scale <num>` | `1.08` | Dot size |
| `--json` | off | JSON result on stdout |

Rendering takes 7 to 8 seconds per second of video on Apple Silicon.

## Making a bust

### `import-glb` (recommended)

Rigs a 3D head (`.glb`, `.gltf` or `.obj`), for example from Tripo P2 Multi View on Scenario. It normalizes the mesh, finds the face orientation with MediaPipe, casts a 320 x 340 grid onto the nearest surface, bakes texture luminance and edges, and computes the jaw, lip, eyelid, brow and neck weights.

```bash
./holome import-glb --glb head.glb --out ~/HoloMe/jane-doe/bust [--yaw 90]
```

`--out` is required. `landmarksDetected: false` in the JSON means the face was not found; set `--yaw` to 90, 180 or 270. For Tripo, set `faceLimit: 50000`: the adaptive default returns a few thousand faces and loses facial detail at the same price.

### `build-bust` (no 3D model needed)

Builds a bust from a FRONT/LEFT/RIGHT character sheet with Depth Anything V2 and MediaPipe, or from three separate photos (`--front`, `--left`, `--right`, framed the same way on plain backgrounds). It needs no credits but gives a flatter face, and long hair becomes a shell around the face.

```bash
./holome build-bust --sheet sheet.png --out ~/HoloMe/jane-doe/bust
```

`--out` is required. It also writes the panels and `qa_panels.png` (see `split-sheet`), and its JSON reports `warnings` and `faceDetailGain`: smooth-skinned faces get an edge-detail lift so they render closer to a textured face.

### `split-sheet`

```bash
./holome split-sheet --sheet sheet.png --out-dir panels
```

Writes `panel_front_hires.png`, `panel_left_hires.png`, `panel_right_hires.png` (Tripo inputs) and `qa_panels.png` (silhouettes in magenta, face points in green), and reports `warnings`. The sheet is split by its panel borders, then by finding the three people, then into equal thirds; force one with `--layout bordered|subjects|thirds`. The LEFT panel's nose must point to the image's left edge and the RIGHT panel's to the right edge: `--swap-profiles` swaps them, and `--mirror-profile left|right` mirrors the good one over the bad one. `build-bust` accepts the same flags.

### `pick-frame`

```bash
./holome pick-frame --video take1.MOV --out frame.jpg
```

Picks the sharpest still that faces the camera with eyes open and mouth closed, at full resolution, and warns when the face is small, dark or never frontal.

### `sheet-template`

```bash
./holome sheet-template --out sheet_template.png
```

Redraws `assets/templates/sheet_template.png`, the empty layout the image model fills.

## Generated talking-head clips (LTX)

For a person with photos but no video, LTX-2.5 on Scenario can generate a talking-head clip from the front panel and an audio file, at up to 20 seconds per clip.

| Command | What it does |
|---|---|
| `./holome split-audio --audio a.wav --out-dir d` | Splits audio at pauses into segments of 19.5 s or less and writes `segments.json` |
| `./holome driving-frame --bust b --panel panels/panel_front_hires.png --out frame.png` | Crops a front image to a 9:16 opening frame centered on the face |
| `./holome join-driving --plan d/segments.json --videos 1.mp4 2.mp4 --out driving.mp4` | Trims each clip to its segment and joins them, locked to the original audio |

Render the joined clip with `--mocap-video driving.mp4`.

## Compositing

- **ProRes 4444 `.mov`:** put it on a track above your footage in Premiere Pro, DaVinci Resolve, Final Cut Pro or After Effects. The alpha keeps the soft glow without a dark fringe.
- **Black-background `.mp4`:** set the layer's blend mode to Screen or Linear Dodge (Add).
- **ffmpeg overlay:**

```bash
ffmpeg -i background.mp4 -i clip.mov \
  -filter_complex "[1:v]scale=720:-1[holo];[0:v][holo]overlay=W-w-40:H-h-40:shortest=1" \
  -map 0:v -map 1:a -c:v libx264 -crf 18 -c:a aac composite.mp4
```
