# Scenario Selfie Hologram

A Claude Code skill that turns a person into a talking 3D hologram. Enroll them once from a few selfies, then turn any selfie video from their phone into a hologram video with their real lip, blink and head motion and their own voice.

All generation runs on [Scenario](https://www.scenario.com) through its MCP. Rigging and rendering run locally with the bundled HoloMe CLI, which writes Apple ProRes 4444 with alpha (`.mov`) for compositing and a black-background H.264 (`.mp4`) for Screen or Add blending.

## Demo

See it in action in the [demo video on X](https://x.com/AIandDesign/status/2123896798137954667).

## How it works

**Enroll (once per person, about 255 Scenario CU)**

1. One to five selfies (or a selfie video, from which the skill picks the best frame) plus an empty FRONT/LEFT/RIGHT template go to GPT Image 2.5 Sunburst, which returns a three-view studio character sheet.
2. `holome split-sheet` cuts the sheet into full-resolution panels and writes a QA image. Claude checks identity, profile directions and silhouettes before anything else is spent.
3. Tripo P2 Multi View turns the three panels into a textured 3D head (50,000 faces).
4. `holome import-glb` finds the face with MediaPipe and rigs the head: jaw, lips, eyelids, brows, head and neck.

**Perform (any number of times, no credits)**

5. `holome render --mocap-video <selfie video>` tracks 52 face blendshapes and head pose in every frame, times the lips to the speech with Whisper phonemes, uses the video's audio as the voice, and renders the hologram in headless Chrome with Three.js.

Audio-only clips (`--audio`) and LTX-2.5 generated talking-head clips for people without a video are also supported.

## Requirements

- [Claude Code](https://claude.com/claude-code) with the Scenario MCP connected (see below) and a Scenario account with credits
- macOS on Apple Silicon (tested), with Google Chrome, Chromium or Brave
- Node.js 18 or newer with npm, ffmpeg with ffprobe, curl
- [uv](https://docs.astral.sh/uv/) or Python 3.12

## Install

```bash
git clone https://github.com/TheMarco/Scenario-Selfie-Hologram.git ~/.claude/skills/scenario-selfie-hologram
~/.claude/skills/scenario-selfie-hologram/tool/setup.sh
```

`setup.sh` installs about 1.6 GB into `tool/`: a Python 3.12 virtualenv, Node packages, the MediaPipe face model, the renderer bundle, and the Depth Anything and Whisper weights. It finishes by running `tool/holome doctor`. Pass `--no-prefetch` to download the model weights on first use instead.

### Connect the Scenario MCP

```bash
claude mcp add --transport http --scope user scenario https://mcp.scenario.com/mcp
```

Then run `/mcp` inside Claude Code and sign in to Scenario when prompted.

### Update

```bash
git -C ~/.claude/skills/scenario-selfie-hologram pull
~/.claude/skills/scenario-selfie-hologram/tool/setup.sh
```

Run `setup.sh` again after every pull to pick up new dependencies.

## Use

Start a new Claude Code session after installing. Ask in plain language, or type `/scenario-selfie-hologram` to start the skill directly:

> Enroll me as a hologram from these selfies: ~/Desktop/front.jpg ~/Desktop/left.jpg ~/Desktop/right.jpg

> Turn ~/Movies/intro.MOV into a vertical hologram video of me.

Each person gets a folder in `~/HoloMe/<name>/` with the photos, sheet, 3D head, rigged bust, previews and renders. Once someone is enrolled, every later video of them goes straight to rendering. A video only drives the motion: the hologram always shows the enrolled person.

### Recording tips

- **Selfies:** one straight into the camera, plus one turned partly left and one partly right if you can. Even light, no sunglasses or hat, face sharp.
- **Videos:** front camera at eye level, face and shoulders in frame, light in front of you, quiet room. Start with two seconds of neutral face with your mouth closed, then talk. Any length.

## Cost

| Step | Scenario credits |
|---|---|
| Character sheet (GPT Image 2.5 Sunburst, 2048 x 1152, high) | about 13 CU per attempt |
| 3D head (Tripo P2 Multi View, 50,000 faces, detailed texture) | about 240 CU |
| Hologram video from a selfie video or audio | none, runs locally |
| LTX-2.5 talking-head clip (optional, for people without a video) | about 18 CU per second of audio at 720p |

The skill states the enrollment cost and confirms once before the first billed call.

## Troubleshooting

- `~/.claude/skills/scenario-selfie-hologram/tool/holome doctor` lists every dependency and what is missing.
- Scenario calls fail with an authentication error: run `/mcp` and sign in again.
- The sheet or 3D head looks wrong: the skill regenerates once on its own; better selfies (sharp, even light, one straight-on) fix most cases.

## Limits

- The side views are generated from the photos, so ears, jaw line and the back of the head are the image model's best guess. Turned photos reduce the guessing.
- Thin details such as glasses frames can get lost in the 3D head.
- Smooth skin renders dimmer than a bearded or textured face.
- Tested on macOS with Apple Silicon. Linux should work with Chrome or Playwright Chromium but is untested.

## Repository layout

```
SKILL.md                  the skill Claude Code loads
tool/holome               CLI entry point
tool/setup.sh             installer
tool/cli/                 Node CLI
tool/renderer/            Three.js hologram renderer
tool/scripts/             Python pipeline (frame picking, sheet splitting, rigging, mocap, audio)
tool/assets/templates/    empty character sheet template
```

See [`tool/README.md`](tool/README.md) for every CLI command and flag.

## License

MIT. See [LICENSE](LICENSE).
