---
name: scenario-selfie-hologram
description: Turn a person into a talking 3D hologram. Enroll them once from a few selfies (Scenario makes a FRONT/LEFT/RIGHT character sheet with GPT Image and a 3D head with Tripo P2 Multi View, then HoloMe rigs it locally), then convert any selfie video from their phone into a hologram video that follows their real lip, blink and head motion and keeps their voice. Outputs ProRes 4444 with alpha plus a black-background MP4. Use when the user wants a hologram, holographic avatar, holographic talking head or "hologram of me", sends selfies or a selfie video for one, or wants another hologram clip of someone already enrolled.
---

# Scenario selfie hologram

This skill's base directory is printed above these instructions when the skill loads. In the commands below, `HOLOME` means `<base directory>/tool/holome` and `TOOL` means `<base directory>/tool`. Every HoloMe command accepts `--json` and then prints one JSON object on stdout (progress goes to stderr).

There are two phases:

1. **Enroll (once per person, about 255 Scenario CU):** selfies become a character sheet, the sheet becomes a 3D head, and the head is rigged into a hologram bust.
2. **Perform (any number of times, no credits):** a selfie video of that person drives the bust. The video's audio is the hologram's voice.

Per-person files live in `~/HoloMe/<slug>/` (lowercase name with hyphens, such as `jane-doe`) unless the user names another folder:

```
~/HoloMe/<slug>/
  photos/      reference selfies (and the frame picked from a video)
  sheet.png    generated character sheet
  panels/      full-resolution FRONT/LEFT/RIGHT panels and qa_panels.png
  head.glb     Tripo 3D head
  bust/        rigged hologram bust (bust.bin, bust_meta.json)
  previews/    QA frames
  videos/      selfie videos to perform
  renders/     finished hologram videos
```

If `~/HoloMe/<slug>/bust/bust.bin` exists, the person is enrolled: go straight to the perform phase unless the user wants a new likeness. A video only drives the motion; the hologram always shows the enrolled person, so a video of someone else needs that person enrolled first.

## What to ask the user for

If the user has not sent inputs yet, ask for them with this guidance.

**Selfies for enrollment (1 to 5 photos):** one looking straight into the camera is required; one with the head turned partly left and one partly right improve the side views. Even light on the face, no sunglasses or hat, hair as they usually wear it, face sharp and filling a good part of the frame. JPEG or PNG; convert HEIC/HEIF/WEBP first (`sips -s format jpeg in.heic --out out.jpg` on macOS, or `ffmpeg -i in.webp out.jpg`). A selfie video alone also works: the skill picks its best frame.

**Selfie video to perform:** front camera, phone at eye level at arm's length or on a stand, face and shoulders in frame and roughly facing the lens, light in front of the face rather than behind, a quiet room. Start with two seconds of neutral face with the mouth closed, then speak. Any length; `.MOV` from an iPhone is fine.

If a photo shows several people, ask which one.

## Enroll

### E0: check the install

Run `HOLOME doctor --json`. If `status` is not `ok`, run `TOOL/setup.sh` (the first run installs about 1.6 GB of Python packages, Node packages and model weights into `TOOL`), then run `doctor` again. Setup needs node 18+, npm, ffmpeg, curl, and uv or python3.12; tell the user which one is missing if setup stops.

### E1: pick the Scenario scope and confirm the cost

Call `teams_list`. With one team and one project, use them. With more, ask the user once which to use. Pass `team_id` and `project_id` on every Scenario call. If a billed call is refused for a missing `task_id`, ask the user which task to bill and reuse it for this person.

Before the first billed call, tell the user that enrolling costs about 255 CU (character sheet about 13 CU per attempt, 3D head about 240 CU) and confirm once.

### E2: gather the reference photos

Copy the user's selfies into `photos/`. If they only gave a video, or to add a reference from one:

```bash
HOLOME pick-frame --video <video> --out ~/HoloMe/<slug>/photos/video_frame.jpg --json
```

It picks the sharpest frame that faces the camera with eyes open and mouth closed. Read the frame. Relay any `warnings` (face too small, too dark, never frontal) and ask for better material if the face is not clearly visible.

### E3: upload

Upload with the multipart flow, never inline base64:

1. `upload_asset` with `file_name`, `content_type`, `kind` and `file_size` (from `wc -c < file`).
2. For each returned part: `curl -fS -X PUT -T <file> '<upload_url>'`, using the URL exactly as returned, in single quotes, with no extra headers.
3. `upload_asset_complete` with the `upload_id`. It returns the `asset_id`.

Upload up to five reference photos and `TOOL/assets/templates/sheet_template.png` (an empty three-panel layout with no person in it). Reuse the template across people in the same project: keep its asset id in `~/HoloMe/scenario.json` as `{"<project_id>": {"templateAssetId": "..."}}`.

### E4: generate the character sheet

`model_run` with `model_id: "model_openai-gpt-image-2-5-sunburst"` (about 13 CU). `referenceImages` lists the photos first and the template last. Use the prompt verbatim:

```json
{
  "referenceImages": ["<photo asset_id>", "<more photo asset_ids, optional>", "<template asset_id>"],
  "width": 2048,
  "height": 1152,
  "quality": "high",
  "background": "opaque",
  "numOutputs": 1,
  "prompt": "The last image is an empty character sheet layout. All other images are photos of one person, taken at different moments or from different angles.\n\nFill the layout with three photorealistic studio photos of that person, and keep the layout exactly: white page, the title, the thin grey panel borders, the white gutters between panels, and the labels FRONT, LEFT and RIGHT under the panels. Do not add any other text.\n\nIdentity: the same person in all three panels. Use every photo to get the likeness right: face shape, skin tone, age, eye color, hairstyle and hair length, facial hair, glasses and clothing. Do not beautify or change them.\n\nFraming, identical in all three panels: head and upper chest, the body cut off by the bottom border of the panel. The top of the head sits a little below the top border. The head is the same size and at the same height in all three panels. Long hair stays as it is and may fall onto the shoulders.\n\nFRONT panel: facing the camera straight on, head level, eyes open and looking into the lens, neutral relaxed expression, mouth closed.\nLEFT panel: exact 90 degree side profile, the nose points to the LEFT edge of the image.\nRIGHT panel: exact 90 degree side profile, the nose points to the RIGHT edge of the image.\n\nLighting and background: soft even studio light with no harsh shadows, a plain pure white background inside every panel, nothing else in the panels."
}
```

If the job returns `in_progress`, call `jobs_wait`. If the provider refuses the image, tell the user and stop.

Call `asset_display` and check: the same person as the photos in all three panels; FRONT straight into the camera; LEFT nose pointing left, RIGHT nose pointing right; heads the same size and height on plain white. If it fails, regenerate once (another 13 CU) and say so; if that fails too, show both and ask. When it passes, `asset_download` with `format: "png"` and save it: `curl -fsSL -o ~/HoloMe/<slug>/sheet.png '<url>'`.

### E5: split the sheet and check it

```bash
HOLOME split-sheet --sheet ~/HoloMe/<slug>/sheet.png --out-dir ~/HoloMe/<slug>/panels --json
```

Read `qaPreview` (magenta lines are silhouettes, green dots are detected face points) and the JSON `warnings`:

| Problem | Fix |
|---|---|
| Profiles in the wrong slots (LEFT panel's nose points right, RIGHT panel's points left) | rerun with `--swap-profiles` |
| One profile wrong (3/4 view, wrong direction), the other good | rerun with `--mirror-profile <good side>` (`left` or `right`) |
| Silhouette leaks into the background or cuts through the head | regenerate the sheet (E4) |
| `front: face is turned`, or green dots off the eyes, nose and mouth | regenerate the sheet (E4) |
| Panels split in the wrong places | rerun with `--layout subjects` or `--layout thirds` |

### E6: make the 3D head on Scenario

Upload `panels/panel_front_hires.png`, `panel_left_hires.png` and `panel_right_hires.png` (E3 flow), then `model_run` with `model_id: "model_tripo-p2-multiview-to-3d"` (about 240 CU, 2 to 3 minutes; call `jobs_wait` until done):

```json
{
  "frontImage": "<front panel asset_id>",
  "leftImage": "<left panel asset_id>",
  "rightImage": "<right panel asset_id>",
  "texture": true,
  "textureQuality": "detailed",
  "faceLimit": 50000
}
```

The LEFT panel (nose pointing left) shows the person's own left side, which is what `leftImage` expects. Keep `faceLimit: 50000`: without it Tripo returns a few thousand faces and the face loses its detail, at the same price. `asset_download` the result with `format: "glb"` to `~/HoloMe/<slug>/head.glb`.

### E7: rig the head

```bash
HOLOME import-glb --glb ~/HoloMe/<slug>/head.glb --out ~/HoloMe/<slug>/bust --json
```

It normalizes the mesh, finds the face with MediaPipe, and bakes the rig (jaw, lips, eyelids, brows, head and neck). `landmarksDetected` must be `true`. If it is `false`, rerun with `--yaw <degrees>` (90, 180 or 270) until it is.

Fallback without the 3D step (no credits, lower quality: long hair turns into a shell around the face and the face is flatter): `HOLOME build-bust --sheet ~/HoloMe/<slug>/sheet.png --out ~/HoloMe/<slug>/bust --json`. Use it only when the user declines the 240 CU or Tripo fails twice.

### E8: preview frames

```bash
HOLOME render --bust ~/HoloMe/<slug>/bust --out ~/HoloMe/<slug>/previews/hero.png --preview-frame 0 --camera hero --json
HOLOME render --bust ~/HoloMe/<slug>/bust --out ~/HoloMe/<slug>/previews/talk.png --preview-frame 0 --camera front --preview-jaw 0.7 --json
```

Read both. The hologram should show the person's head shape and hair, and the second frame an open mouth in the right place. Tell the user enrollment is done and what it cost.

## Perform

### P1: selfie video to hologram video (default)

```bash
HOLOME render --bust ~/HoloMe/<slug>/bust --mocap-video <video> --out ~/HoloMe/<slug>/renders/<clip-name> --format both --json
```

The renderer tracks the face in every frame (52 blendshapes plus head pose), times the lips to the speech with Whisper phonemes, and uses the video's audio track as the voice. Add `--audio <file>` to replace the voice with cleaner audio of the same length, and `--text "<transcript>"` when the words are known (names and brand words align better). `--format both` writes `<clip-name>.mov` (ProRes 4444 with alpha) and `<clip-name>.mp4` (black background, for Screen or Add blending). Rendering takes 7 to 8 seconds per second of video on Apple Silicon.

Check the result: extract a frame with `ffmpeg -ss 2 -i <clip>.mp4 -frames:v 1 <frame>.png` and Read it. The render log on stderr reports tracked frames out of total frames; if many were lost, tell the user what to change in the recording.

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

### P2: audio only

With an audio file and no video, use `--audio <file>` instead of `--mocap-video`. Lips follow the speech; blinks and head motion are procedural.

### P3 (only on request): generated talking-head video

For a person with photos but no video who wants video-driven motion, LTX-2.5 can generate a talking-head clip from the front panel and the audio. It costs about 18 CU per second of audio at 720p. Quote the total with `dry_run: true` per segment and ask before running.

1. `HOLOME split-audio --audio <audio> --out-dir ~/HoloMe/<slug>/driving/audio --json` (segments of 19.5 s or less; LTX accepts 20 s at most).
2. `HOLOME driving-frame --bust ~/HoloMe/<slug>/bust --panel ~/HoloMe/<slug>/panels/panel_front_hires.png --out ~/HoloMe/<slug>/driving/frame.png --json`.
3. Upload the frame and each segment WAV (`kind: "audio"`, `content_type: "audio/wav"`), then per segment `model_run` with `model_id: "model_ltx-2-5-fast-audio-to-video"`, `wait: false`, and `jobs_wait` on all jobs:

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

4. Download the clips in order and join them: `HOLOME join-driving --plan ~/HoloMe/<slug>/driving/audio/segments.json --videos <clips in order> --out ~/HoloMe/<slug>/driving/driving.mp4 --json`.
5. Render as in P1 with `--mocap-video ~/HoloMe/<slug>/driving/driving.mp4 --audio <original audio>`.

## Report

Give the user the output paths, the Scenario credits spent (sheet attempts, 3D head, any LTX segments), any QA fixes applied, and any warnings that remain.

## Limits

- The side views are generated from the photos, so ears, jaw line and the back of the head are the image model's best guess. Turned photos reduce the guessing.
- Thin details such as glasses frames can get lost in the 3D head.
- Smooth skin renders dimmer than a bearded or textured face.
- Tested on macOS with Apple Silicon and Google Chrome. Linux should work with Chrome or Playwright Chromium but is untested.
