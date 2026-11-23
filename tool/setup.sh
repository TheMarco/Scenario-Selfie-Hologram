#!/usr/bin/env bash
set -euo pipefail

# Script: setup.sh
# Purpose: Install everything HoloMe needs next to this script: a Python 3.12 virtualenv,
#          Node packages, the MediaPipe face model, the renderer bundle and (by default)
#          the Depth Anything and Whisper weights.
# Requirements: node + npm, ffmpeg + ffprobe, curl, and uv or python3.12.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANDMARKER_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

REQUIRED_ARGS=()
REQUIRED_ENV_VARS=()
REQUIRED_PROGRAMS=("node" "npm" "ffmpeg" "ffprobe" "curl")

PREFETCH=1
PYTHON_OVERRIDE=""

usage() {
  cat << 'EOF'
Usage: setup.sh [--no-prefetch] [--python <path-to-python3.12>]

Installs HoloMe into the folder that contains this script:
  .venv/                 Python 3.12 environment (requirements.txt)
  node_modules/          Node packages (package-lock.json)
  assets/models/         MediaPipe face_landmarker.task
  renderer/dist/         Renderer bundle

Options:
  --no-prefetch   Skip downloading Depth Anything V2 Small and Whisper base.en now
                  (they download on first use instead, about 250 MB together)
  --python PATH   Python 3.12 interpreter to build the virtualenv from
  -h, --help      Show this help

External tools:
  node (18 or newer) and npm, ffmpeg and ffprobe, curl,
  and either uv (https://docs.astral.sh/uv/) or python3.12.
  Google Chrome, Chromium or Brave is used for rendering when installed.

Examples:
  ./setup.sh
  ./setup.sh --no-prefetch
EOF
}

check_requirements() {
  local -r provided_arg_count=$1
  local missing=0

  if [ ${#REQUIRED_ARGS[@]} -gt 0 ] && [ "$provided_arg_count" -lt ${#REQUIRED_ARGS[@]} ]; then
    printf 'Error: Expected %s arguments (%s) but received %s.\n' \
      ${#REQUIRED_ARGS[@]} "${REQUIRED_ARGS[*]}" "$provided_arg_count" >&2
    missing=1
  fi

  local env_var
  if [ ${#REQUIRED_ENV_VARS[@]} -gt 0 ]; then
    for env_var in "${REQUIRED_ENV_VARS[@]}"; do
      if [ -z "${!env_var:-}" ]; then
        printf 'Error: Missing required environment variable %s. Please set it before rerunning.\n' "$env_var" >&2
        missing=1
      fi
    done
  fi

  local program
  for program in "${REQUIRED_PROGRAMS[@]}"; do
    if ! command -v "$program" > /dev/null 2>&1; then
      printf 'Error: Required program %s is not installed or not on PATH. Please install it first.\n' "$program" >&2
      missing=1
    fi
  done

  if [ -n "$PYTHON_OVERRIDE" ]; then
    if [ ! -x "$PYTHON_OVERRIDE" ]; then
      printf 'Error: --python %s is not an executable file.\n' "$PYTHON_OVERRIDE" >&2
      missing=1
    fi
  elif ! command -v uv > /dev/null 2>&1 && ! command -v python3.12 > /dev/null 2>&1; then
    printf 'Error: Neither uv nor python3.12 was found. Please install uv (https://docs.astral.sh/uv/) or Python 3.12.\n' >&2
    missing=1
  fi

  if command -v node > /dev/null 2>&1; then
    local node_major
    node_major="$(node -p 'process.versions.node.split(".")[0]')"
    if [ "$node_major" -lt 18 ]; then
      printf 'Error: Node.js 18 or newer is required (found %s). Please upgrade Node.\n' "$(node --version)" >&2
      missing=1
    fi
  fi

  if [ "$missing" -ne 0 ]; then
    printf '\n' >&2
    usage >&2
    return 1
  fi
}

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --no-prefetch)
        PREFETCH=0
        ;;
      --python)
        if [ $# -lt 2 ]; then
          printf 'Error: --python needs a path.\n\n' >&2
          usage >&2
          exit 1
        fi
        PYTHON_OVERRIDE="$2"
        shift
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      *)
        printf 'Error: Unknown option %s.\n\n' "$1" >&2
        usage >&2
        exit 1
        ;;
    esac
    shift
  done
}

log() {
  printf '[setup] %s\n' "$*" >&2
}

setup_python() {
  local venv_python="$ROOT_DIR/.venv/bin/python"
  if [ ! -x "$venv_python" ]; then
    if [ -n "$PYTHON_OVERRIDE" ]; then
      log "Creating .venv from $PYTHON_OVERRIDE"
      "$PYTHON_OVERRIDE" -m venv "$ROOT_DIR/.venv"
    elif command -v uv > /dev/null 2>&1; then
      log "Creating .venv with uv (Python 3.12)"
      uv venv --python 3.12 "$ROOT_DIR/.venv"
    else
      log "Creating .venv with python3.12"
      python3.12 -m venv "$ROOT_DIR/.venv"
    fi
  else
    log "Reusing existing .venv"
  fi

  log "Installing Python packages (torch is large; the first install takes a few minutes)"
  if command -v uv > /dev/null 2>&1; then
    uv pip install --python "$venv_python" -r "$ROOT_DIR/requirements.txt"
  else
    "$venv_python" -m pip install --upgrade pip
    "$venv_python" -m pip install -r "$ROOT_DIR/requirements.txt"
  fi
}

setup_node() {
  log "Installing Node packages"
  if [ -f "$ROOT_DIR/package-lock.json" ]; then
    (cd "$ROOT_DIR" && npm ci --no-audit --no-fund)
  else
    (cd "$ROOT_DIR" && npm install --no-audit --no-fund)
  fi
  log "Building renderer bundle"
  node "$ROOT_DIR/renderer/bundle.mjs"
}

setup_models() {
  local model_dir="$ROOT_DIR/assets/models"
  local landmarker="$model_dir/face_landmarker.task"
  mkdir -p "$model_dir"
  if [ ! -s "$landmarker" ]; then
    log "Downloading MediaPipe face_landmarker.task"
    curl -fsSL -o "$landmarker.part" "$LANDMARKER_URL"
    mv "$landmarker.part" "$landmarker"
  else
    log "face_landmarker.task already present"
  fi

  if [ ! -s "$ROOT_DIR/assets/templates/sheet_template.png" ]; then
    log "Writing character sheet template"
    "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/make_sheet_template.py" \
      --out "$ROOT_DIR/assets/templates/sheet_template.png" > /dev/null
  fi

  if [ "$PREFETCH" -eq 1 ]; then
    log "Prefetching Depth Anything V2 Small and Whisper base.en weights"
    "$ROOT_DIR/.venv/bin/python" - << 'PY'
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from faster_whisper import WhisperModel
model_id = "depth-anything/Depth-Anything-V2-Small-hf"
AutoImageProcessor.from_pretrained(model_id)
AutoModelForDepthEstimation.from_pretrained(model_id)
WhisperModel("base.en", device="cpu", compute_type="int8")
print("weights ready")
PY
  fi
}

setup_browser() {
  local candidate
  for candidate in \
    "${HOLOME_CHROME_PATH:-}" \
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    "/Applications/Chromium.app/Contents/MacOS/Chromium" \
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" \
    "/usr/bin/google-chrome" \
    "/usr/bin/chromium"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
      log "Using browser: $candidate"
      return 0
    fi
  done
  log "No Chrome/Chromium found; installing Playwright Chromium"
  (cd "$ROOT_DIR" && npx --yes playwright-core install chromium)
}

main() {
  parse_args "$@"
  check_requirements "$#" || exit 1

  chmod +x "$ROOT_DIR/holome"
  setup_python
  setup_node
  setup_models
  setup_browser

  log "Checking the installation"
  "$ROOT_DIR/holome" doctor
}

main "$@"
