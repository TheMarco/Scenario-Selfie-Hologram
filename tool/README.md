# HoloMe

High-end, audio-driven 3D hologram generator CLI built for video editors and autonomous agentic video pipelines.

Reconstructs a sculpted, anatomically rigged 3D bust from a 3-view character sheet (Front / Left / Right) and renders 1080p/4K holographic talking-head videos driven directly by spoken-word audio — outputting both **Apple ProRes 4444 (`.mov` with 10-bit alpha)** and **black-background H.264 (`.mp4` for Screen/Add blending)** in a single pass.

## Quick Start

```bash
# Render BOTH transparent ProRes 4444 (.mov) AND black-background H.264 (.mp4)
./holome render \
  --audio path/to/speech.wav \
  --out outputs/my_clip

# Fast 1-frame PNG preview (~1.5s) for framing & style QA
./holome render \
  --audio path/to/speech.wav \
  --out outputs/preview.png \
  --preview-frame 1.5 \
  --camera hero

# Clean overlay (no emitter base, transparent alpha only) for picture-in-picture / compositing
./holome render \
  --audio path/to/speech.wav \
  --out outputs/overlay.mov \
  --format mov \
  --no-emitter \
  --camera front

# Agentic pipeline mode (strict JSON metadata on stdout, progress logs on stderr)
./holome render \
  --audio speech.mp3 \
  --text "Optional spoken transcript for tighter phoneme lip-sync." \
  --out outputs/scene_01 \
  --format both \
  --json
```

## CLI Commands & Key Flags

### `holome render`
| Flag | Default | Description |
|---|---|---|
| `-a, --audio <path>` | *(required for video)* | Any spoken-word audio file (`.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`) |
| `-d, --driving-video <path>` | — | Optional live-action talking-head video (from Scenario `LTX-2.5 Fast` or a webcam/iPhone clip) projected live frame-by-frame onto the 3D bust for cinema-grade lip-sync, teeth, mustache flex & blinks |
| `-o, --out <path>` | *(required)* | Output file or stem (with `--format both`, writes `<stem>.mov` and `<stem>.mp4`) |
| `-f, --format <type>` | `both` | `both` (ProRes 4444 `.mov` + black `.mp4`), `mov`, `mp4`, `webm` (VP9 alpha), or `png` |
| `-r, --resolution <res>` | `1080p` | `720p`, `1080p`, `4k`, `vertical` (`1080x1920`, auto-switches to 9:16 top/bottom social HUD layout), `square` (`1080x1080`), or `WxH` |
| `--fps <num>` | `30` | `24`, `30`, or `60` |
| `--camera <preset>` | `hero` | `hero` (3/4 angle), `front`, `left34`, `right34`, `orbit` |
| `--camera-motion <mode>` | `subtle` | `subtle` (organic drift), `static`, `orbit` |
| `--camera-yaw <deg>` | — | Custom horizontal camera angle in degrees |
| `--camera-pitch <deg>` | — | Custom vertical camera angle in degrees |
| `--camera-zoom <num>` | `1.0` | Camera zoom multiplier (e.g. `1.2` closer, `0.85` wider) |
| `--look-at-camera` / `--no-look-at-camera` | `--look-at-camera` | Aim head & eye gaze directly at the camera lens |
| `--hud` / `--no-hud` | `--hud` | Toggle floating 3D sci-fi HUD telemetry (grids, reticles, live audio equalizer) |
| `--emitter` / `--no-emitter` | `--emitter` | Toggle circular holo-projector base & upward volumetric rays |
| `--background <mode>` | `auto` | `auto`, `transparent`, `black`, `lab` (dark volumetric backdrop) |
| `--color <spec>` | `cyan` | Presets (`cyan`, `ice`, `amber`, `emerald`, `crimson`, `violet`) or `#RRGGBB` |
| `--bloom <num>` | `1.0` | Volumetric glow strength (`0.0`–`2.0`) |
| `--glitch <num>` | `0.35` | Audio-reactive holographic glitch intensity (`0.0`–`1.0`) |
| `--text <string>` | — | Optional spoken transcript hint for Whisper word-level ARPABET phoneme alignment (auto-detected even without `--text`) |
| `--transcript <path>` | — | Optional `.txt` file containing the spoken transcript |
| `--bust <dir>` | `assets/bust` | Bust asset folder made by `build-bust` or `import-glb`. Use one folder per person. |
| `--preview-frame <sec>` | — | Render a single PNG frame at timestamp `<sec>` (~1.5s) |
| `--json` | `false` | Emit structured JSON metadata to stdout |

### `holome build-bust`
Builds a sculpted and rigged 3D bust from a 3-panel FRONT/LEFT/RIGHT character sheet, or from three separate photos. `--out` is required, so a build never overwrites another person's bust by accident.
```bash
./holome build-bust --sheet inputs/character_sheet.jpg --out assets/bust
./holome build-bust --front f.png --left l.png --right r.png --out ~/HoloMe/jane/bust
```

The sheet is split with bordered panel detection first, then by finding the three people (`subjects`), then into equal thirds. Force one with `--layout bordered|subjects|thirds`. In the LEFT panel the nose must point to the image's left edge, in the RIGHT panel to the right edge. If a sheet gets that wrong, `--swap-profiles` swaps the two profiles and `--mirror-profile left|right` mirrors the good one.

Every build writes `qa_panels.png` (silhouettes in magenta, face points in green) and reports `warnings` and `faceDetailGain` in its JSON. Panels are resampled to 468 px tall before baking, and smooth-skinned faces get an edge-detail lift so they render about as bright as a bearded one. Mesh resolution defaults to 320 x 340, the resolution of the bundled busts.

### Selfie to hologram (agent skill)
`skill/SKILL.md` is a Claude Code skill that turns one selfie into a hologram through the Scenario MCP: the selfie and `assets/templates/sheet_template.png` go to GPT Image 2.5 Sunburst to make the character sheet, then `build-bust` and `render` run locally. Build the installable skill with:
```bash
scripts/package_skill.sh                # writes dist/scenario-selfie-hologram/ and dist/scenario-selfie-hologram.zip
cp -R dist/scenario-selfie-hologram ~/.claude/skills/
~/.claude/skills/scenario-selfie-hologram/tool/setup.sh
```
The package holds SKILL.md and a clean copy of this tool. It contains no busts, inputs or outputs.

### Helper commands
| Command | What it does |
|---|---|
| `holome doctor` | Checks Node packages, the Python environment, the face model, ffmpeg and Chrome |
| `holome sheet-template --out t.png` | Writes the empty FRONT/LEFT/RIGHT sheet layout used as the image model's reference |
| `holome split-audio --audio a.wav --out-dir d` | Splits audio at pauses into segments of 19.5 s or less for LTX (20 s cap), writes `segments.json` |
| `holome driving-frame --bust b --out f.png` | Crops the bust's full-resolution front panel into a 9:16 opening frame for LTX |
| `holome join-driving --plan segments.json --videos 1.mp4 2.mp4 --out d.mp4` | Joins per-segment LTX clips, trimmed to each segment, locked to the original audio |

`setup.sh` installs the Python environment (uv or python3.12), Node packages, the MediaPipe model and the renderer bundle into this folder, and prefetches the Depth Anything and Whisper weights.

### `holome import-glb` (Scenario.com / Custom 3D Bust Support)
If you generate a high-res sculpted 3D bust on **Scenario.com** (or Meshy / Rodin / Blender) and export it as `.glb`, you can plug it directly into the HoloMe shader & lip-sync pipeline:
```bash
./holome import-glb --glb path/to/scenario_bust.glb --out assets/bust_scenario
./holome render --bust assets/bust_scenario --audio speech.wav --out outputs/scenario_clip
```

## Compositing in Video Editors & FFmpeg

- **ProRes 4444 (`.mov`)**: Drop directly onto a track above your footage in Premiere Pro, DaVinci Resolve, Final Cut Pro, or After Effects. The optical straight alpha preserves soft glowing cyan bloom halos with zero dark fringe.
- **Black-Background (`.mp4`)**: Set layer blend mode to **Screen** or **Linear Dodge (Add)** for instant luminous holographic compositing.
- **FFmpeg Overlay (Agentic Pipelines)**:
  ```bash
  ffmpeg -i background.mp4 -i outputs/my_clip.mov \
    -filter_complex "[1:v]scale=720:-1[holo];[0:v][holo]overlay=W-w-40:H-h-40:shortest=1" \
    -map 0:v -map 1:a -c:v libx264 -crf 18 -c:a aac final_composite.mp4
  ```

## Agentic Pipeline Integration

See [`AGENT_SKILL.md`](./AGENT_SKILL.md) for drop-in instructions you can give Claude or other video-creation agents.
