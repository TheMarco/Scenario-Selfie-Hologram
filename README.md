# Scenario Selfie Hologram

A Claude Code skill that turns one selfie into a talking 3D hologram video of that person, driven by any spoken-word audio.

The skill runs the AI steps through the [Scenario](https://www.scenario.com) MCP and everything else locally with the bundled HoloMe CLI. It produces Apple ProRes 4444 with alpha (`.mov`) for compositing and a black-background H.264 (`.mp4`) for Screen or Add blending.

## How it works

1. The selfie and an empty FRONT/LEFT/RIGHT sheet template (`tool/assets/templates/sheet_template.png`) go to GPT Image 2.5 Sunburst on Scenario, which returns a three-view studio character sheet of the person.
2. Claude checks the sheet for identity, profile directions and framing before anything else runs.
3. `holome build-bust` splits the sheet, runs Depth Anything V2 and MediaPipe Face Landmarker, and builds a rigged point-cloud bust (jaw, lips, eyelids, brows, head and neck). It writes a QA image that Claude inspects.
4. `holome render` analyzes the audio (faster-whisper word timing plus CMUdict phonemes) and renders the hologram in headless Chrome with Three.js.

An optional driving-video mode generates an LTX-2.5 talking-head clip of the person on Scenario and projects it onto the bust. It costs more credits and is off unless you ask for it.

## Requirements

- [Claude Code](https://claude.com/claude-code) with the Scenario MCP connected
- macOS on Apple Silicon (tested), with Google Chrome, Chromium or Brave
- Node.js 18 or newer with npm, ffmpeg with ffprobe, curl
- [uv](https://docs.astral.sh/uv/) or Python 3.12

## Install

```bash
git clone https://github.com/TheMarco/Scenario-Selfie-Hologram.git ~/.claude/skills/scenario-selfie-hologram
~/.claude/skills/scenario-selfie-hologram/tool/setup.sh
```

`setup.sh` installs about 1.6 GB into `tool/`: a Python 3.12 virtualenv, Node packages, the MediaPipe face model, the renderer bundle, and the Depth Anything and Whisper weights. It finishes by running `tool/holome doctor`. Pass `--no-prefetch` to download the model weights on first use instead.

## Use

Ask Claude in plain language, for example:

> Make a hologram of me from ~/Desktop/selfie.jpg talking with ~/Desktop/intro.m4a. Vertical, for Reels.

Each person gets a folder in `~/HoloMe/<name>/` with the selfie, the generated sheet, the bust, previews and renders. Later clips for the same person reuse the bust and cost no Scenario credits.

## Cost

| Step | Scenario credits |
|---|---|
| Character sheet (GPT Image 2.5 Sunburst, 2048 x 1152, high) | about 13 CU per attempt |
| Audio-driven render | none, runs locally |
| Driving-video mode (LTX-2.5 Fast, optional) | about 18 CU per second of audio at 720p, 26 at 1080p |

The skill quotes the LTX cost with a dry run and asks before spending it.

## Limits

- The side views are generated from one photo, so ears, jaw line and the back of the head are the image model's guesses.
- The bust shape comes from silhouettes plus depth estimation. Long hair forms a shell around the face, and hats, hoods or large headphones change the head shape.
- Smooth skin renders dimmer than a bearded or textured face. The bust builder lifts edge detail on smooth faces to reduce the gap.
- Tested on macOS with Apple Silicon. Linux should work with Chrome or Playwright Chromium but is untested.

## Repository layout

```
SKILL.md                  the skill Claude Code loads
tool/holome               CLI entry point (render, build-bust, doctor, split-audio, driving-frame, join-driving, ...)
tool/setup.sh             installer
tool/cli/                 Node CLI
tool/renderer/            Three.js hologram renderer
tool/scripts/             Python pipeline (bust builder, audio analysis, driving video)
tool/assets/templates/    empty character sheet template
```

See [`tool/README.md`](tool/README.md) for every CLI command and flag.
