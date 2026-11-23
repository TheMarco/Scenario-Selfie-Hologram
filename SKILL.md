---
name: scenario-selfie-hologram
description: Turn one selfie into a talking 3D hologram video of that person, driven by any spoken-word audio. Uses the Scenario MCP to turn the selfie into a FRONT/LEFT/RIGHT character sheet (and, on request, an LTX talking-head driving video), then the bundled HoloMe CLI to build a rigged 3D point-cloud bust and render ProRes 4444 with alpha plus a black-background MP4. Use when the user wants a hologram, holographic avatar, holographic talking head or "hologram of me" from a photo or selfie and an audio file, or wants another clip rendered for a person who already has a bust.
---

# HoloMe selfie to talking hologram

This skill's base directory is printed above these instructions when the skill loads. In the commands below, `HOLOME` means `<base directory>/tool/holome` and `TOOL` means `<base directory>/tool`. Every HoloMe command accepts `--json` and then prints one JSON object on stdout (progress goes to stderr).

## What the user provides

- A selfie or portrait photo with one person and the face clearly visible. JPEG or PNG. Convert HEIC/HEIF/WEBP first: `sips -s format jpeg in.heic --out selfie.jpg` on macOS, or `ffmpeg -i in.webp selfie.jpg`.
- A spoken-word audio file (wav, mp3, m4a, flac, ogg) and, if they have it, the transcript.
- A name for the person, used for the folder (lowercase, hyphens: `jane-doe`).

If the audio is missing, ask for it. If the photo shows several people, ask which one.

Per-person files live in `~/HoloMe/<slug>/` unless the user names another folder:

```
~/HoloMe/<slug>/
  selfie.jpg        copy of the input photo
  sheet.png         generated character sheet
  bust/             3D bust (bust.bin, bust_meta.json, qa_panels.png, panels)
  previews/         QA preview frames
  driving/          audio segments, LTX clips, driving.mp4 (driving mode only)
  renders/          final videos
```

If `~/HoloMe/<slug>/bust/bust.bin` already exists, skip to step 6 unless the user wants a new likeness.

## Step 0: check the install

Run `HOLOME doctor --json`. If `status` is not `ok`, run `TOOL/setup.sh` (the first run installs about 1.6 GB of Python packages, Node packages and model weights into `TOOL`), then run `doctor` again. Setup needs node 18+, npm, ffmpeg, curl, and uv or python3.12; tell the user which one is missing if setup stops.

## Step 1: pick the Scenario scope

Call `teams_list`. With one team and one project, use them. With more, ask the user once which team and project to use. Pass `team_id` and `project_id` on every Scenario call. If a billed call is refused for a missing `task_id`, ask the user which task to bill and reuse it for this person.

## Step 2: upload the selfie and the sheet template

Upload two local files with the multipart flow, never inline base64:

1. `upload_asset` with `file_name`, `content_type`, `kind: "image"` and `file_size` (from `wc -c < file`).
2. For each returned part: `curl -fS -X PUT -T <file> '<upload_url>'`, using the URL exactly as returned, in single quotes, with no extra headers.
3. `upload_asset_complete` with the `upload_id`. It returns the `asset_id`.

Files: `~/HoloMe/<slug>/selfie.jpg` and `TOOL/assets/templates/sheet_template.png` (an empty three-panel layout with no person in it). The template can be reused for every person in the same project: keep its asset id in `~/HoloMe/scenario.json` as `{"<project_id>": {"templateAssetId": "..."}}` and reuse it when present.

## Step 3: generate the character sheet

Call `model_run` with `model_id: "model_openai-gpt-image-2-5-sunburst"` (about 13 CU) and these parameters, with the prompt verbatim:

```json
{
  "referenceImages": ["<selfie asset_id>", "<template asset_id>"],
  "width": 2048,
  "height": 1152,
  "quality": "high",
  "background": "opaque",
  "numOutputs": 1,
  "prompt": "Image 1 is a photo of a person. Image 2 is an empty character sheet layout.\n\nFill the layout of image 2 with three photorealistic studio photos of the person from image 1, and keep the layout exactly: white page, the title, the thin grey panel borders, the white gutters between panels, and the labels FRONT, LEFT and RIGHT under the panels. Do not add any other text.\n\nIdentity: the same person as image 1 in all three panels. Keep their face shape, skin tone, age, eye color, hairstyle and hair length, facial hair, glasses and clothing. Do not beautify or change them.\n\nFraming, identical in all three panels: head and upper chest, the body cut off by the bottom border of the panel. The top of the head sits a little below the top border. The head is the same size and at the same height in all three panels. Long hair stays as it is and may fall onto the shoulders.\n\nFRONT panel: facing the camera straight on, head level, eyes open and looking into the lens, neutral relaxed expression, mouth closed.\nLEFT panel: exact 90 degree side profile, the nose points to the LEFT edge of the image.\nRIGHT panel: exact 90 degree side profile, the nose points to the RIGHT edge of the image.\n\nLighting and background: soft even studio light with no harsh shadows, a plain pure white background inside every panel, nothing else in the panels."
}
```

The order of `referenceImages` matters: selfie first, template second. If the job returns `in_progress`, call `jobs_wait` until it completes. If the model refuses the image, tell the user the provider declined it and stop.

## Step 4: check the sheet, then download it

Call `asset_display` on the result and look at it. It passes when all of these hold:

- All three panels show the same person as the selfie (face shape, hair, glasses, facial hair).
- FRONT looks straight into the camera. LEFT has the nose pointing left, RIGHT has the nose pointing right.
- Heads are the same size and height in all panels, on a plain white background, with nothing else in the panels.

If it fails, regenerate once with the same call (another 13 CU) and say so. If the second attempt also fails, show both to the user and ask how to proceed. When it passes, `asset_download` with `format: "png"` and save it with `curl -fsSL -o ~/HoloMe/<slug>/sheet.png '<url>'`.

## Step 5: build the bust and read its QA preview

```bash
HOLOME build-bust --sheet ~/HoloMe/<slug>/sheet.png --out ~/HoloMe/<slug>/bust --json
```

Takes 5 to 15 seconds. Then Read `qaPreview` (`bust/qa_panels.png`): magenta lines are the silhouettes and green dots are the detected face points. Check the JSON `warnings` and the image:

| Problem | Fix |
|---|---|
| Profiles are in the wrong slots (LEFT panel's nose points right and RIGHT panel's points left) | rebuild with `--swap-profiles` |
| One profile is wrong (3/4 view, wrong direction, broken silhouette), the other is good | rebuild with `--mirror-profile <good side>` (`left` or `right`) |
| Silhouette leaks into the background or cuts through the head, or a silhouette warning | regenerate the sheet (step 3) |
| `front: face is turned` warning, or green dots off the eyes, nose and mouth | regenerate the sheet (step 3) |
| Panels split in the wrong places | rebuild with `--layout subjects` (finds the three people) or `--layout thirds` |

`No face detected in FRONT panel` means the front panel has no usable face; regenerate the sheet.

## Step 6: preview frames

```bash
HOLOME render --bust ~/HoloMe/<slug>/bust --out ~/HoloMe/<slug>/previews/hero.png --preview-frame 0 --camera hero --json
HOLOME render --bust ~/HoloMe/<slug>/bust --out ~/HoloMe/<slug>/previews/talk.png --preview-frame 0 --camera front --preview-jaw 0.7 --json
```

Read both. The hologram should show the person's head shape, hair and glasses, and the second frame should show an open mouth on the mouth. Long hair renders as a shell around the face and smooth skin renders dimmer than a bearded or textured face; both are expected.

## Step 7: render the video (default: audio-driven, no Scenario credits)

```bash
HOLOME render --bust ~/HoloMe/<slug>/bust --audio <audio> --text "<transcript, if known>" --out ~/HoloMe/<slug>/renders/<clip-name> --format both --json
```

`--format both` writes `<clip-name>.mov` (ProRes 4444 with alpha) and `<clip-name>.mp4` (black background, for Screen or Add blending). Rendering takes 7 to 8 seconds per second of video on Apple Silicon. Map requests to flags:

| Request | Flags |
|---|---|
| Vertical for Shorts, Reels, TikTok | `--resolution vertical` |
| Corner overlay or picture-in-picture | `--format mov --no-emitter --camera front` |
| Fewer sci-fi graphics | `--no-hud` |
| Dark standalone shot | `--format mp4 --background lab` |
| Another color | `--color ice`, `amber`, `emerald`, `crimson`, `violet` or `#RRGGBB` |
| 4K | `--resolution 4k` |
| Closer or wider | `--camera-zoom 1.15` or `--camera-zoom 0.85` |

Run `HOLOME render --help` for the rest.

## Step 8 (only on request): driving-video mode

Use this only when the user asks for lip sync from a generated talking-head video. It projects an LTX-2.5 talking-head video of the person onto the bust. It costs Scenario credits (about 18 CU per second of audio at 720p, 26 at 1080p), and in testing on a smooth-skinned face with long hair it looked no better than the audio-driven render. Quote the cost first:

1. `HOLOME split-audio --audio <audio> --out-dir ~/HoloMe/<slug>/driving/audio --json` splits at pauses into segments of 19.5 seconds or less (LTX accepts 20 seconds at most) and writes `segments.json`.
2. `HOLOME driving-frame --bust ~/HoloMe/<slug>/bust --out ~/HoloMe/<slug>/driving/frame.png --json` makes the 9:16 opening frame from the front panel.
3. Upload the frame once and every segment WAV (`kind: "audio"`, `content_type: "audio/wav"`) with the step 2 flow.
4. For each segment, `model_run` with `model_id: "model_ltx-2-5-fast-audio-to-video"` and `dry_run: true`, add up `creativeUnitsCost`, tell the user the total and ask before running. Then run each segment with `wait: false` and wait on all job ids with `jobs_wait`. Parameters:

```json
{
  "audio": "<segment asset_id>",
  "image": "<frame asset_id>",
  "prompt": "The person in the image speaks directly to the camera, saying the words in the audio. Natural lip movement matching the speech, small natural head movements, occasional blinks, relaxed expression. The camera stays still, the framing and the plain white studio background stay the same.",
  "resolution": "720p",
  "aspectRatio": "9:16",
  "fps": 24,
  "cameraMotion": "static"
}
```

5. `asset_download` each result and save as `driving/ltx/segment_01.mp4`, `segment_02.mp4`, ... in segment order.
6. `HOLOME join-driving --plan ~/HoloMe/<slug>/driving/audio/segments.json --videos <clips in order> --out ~/HoloMe/<slug>/driving/driving.mp4 --json` trims each clip to its segment and locks the result to the original audio.
7. Render as in step 7, adding `--driving-video ~/HoloMe/<slug>/driving/driving.mp4`.

## Step 9: report

Give the user the output paths, the Scenario credits spent (sheet attempts, LTX segments), any QA fixes applied (`--swap-profiles`, `--mirror-profile`, regenerations) and any warnings that remain.

## Limits

- The side views come from one photo, so ears, jaw line and the back of the head are the image model's guesses.
- The bust is built from silhouettes plus depth estimation: hats, hoods, large headphones and very large hair change the head shape.
- Rendering is tested on macOS with Apple Silicon and Google Chrome. Linux should work with Chrome or Playwright Chromium but is untested.
