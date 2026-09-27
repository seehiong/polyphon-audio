#!/usr/bin/env bash
# Record N seconds from the default capture device into a 16kHz mono WAV,
# suitable as --prompt-audio input to persona_turn.py.
#
# Usage: record_mic.sh <output.wav> [seconds] [alsa-device]
set -euo pipefail

out="${1:?usage: record_mic.sh <output.wav> [seconds] [alsa-device]}"
seconds="${2:-5}"
device="${3:-plughw:2,0}"

echo "Recording ${seconds}s from ${device} -> ${out} (speak now)..." >&2
arecord -D "$device" -f S16_LE -r 16000 -c 1 -d "$seconds" "$out"
echo "done." >&2
