#!/usr/bin/env python3
"""Multi-voice narration/dialogue TTS - the "ElevenLabs dialogue mode"
equivalent: authored lines, each spoken by a named voice, no LLM involved.

Useful for audiobook narration or scripted multi-character dialogue: write
the lines yourself (or generate them however you like upstream), tag each
with a speaker, and get one clip per line plus a stitched-together whole.

A line's text may include inline "[pause 0.6s]" markers - each spoken
chunk around a marker is synthesized separately and real silence is
spliced in for the gap, since Chatterbox has no pause token of its own
and would otherwise read the marker aloud literally.

    --voices voices.json:  {"alice": "path/to/alice_ref.wav", "bob": "..."}
    --script script.json:  [{"speaker": "alice", "text": "..."}, ...]

Prints a single JSON line to stdout on success:
{"lines": [...], "full_audio": "..."}
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from persona_turn import (  # noqa: E402
    _DEFAULT_BACKEND,
    _DEFAULT_CLI,
    _DEFAULT_TTS_MODEL,
    require,
    synthesize,
    synthesize_with_pauses,
)


def concat_wavs(paths: list[Path], out_path: Path) -> None:
    with wave.open(str(paths[0]), "rb") as f:
        params = f.getparams()
        frames = [f.readframes(f.getnframes())]
    for p in paths[1:]:
        with wave.open(str(p), "rb") as f:
            if (f.getnchannels(), f.getsampwidth(), f.getframerate()) != (
                params.nchannels,
                params.sampwidth,
                params.framerate,
            ):
                sys.exit(f"{p} has a different audio format than {paths[0]}, cannot concatenate")
            frames.append(f.readframes(f.getnframes()))
    with wave.open(str(out_path), "wb") as out:
        out.setparams(params)
        for fr in frames:
            out.writeframes(fr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voices", required=True, help='JSON voice registry: {"id": "ref.wav", ...}')
    parser.add_argument("--script", required=True, help='JSON lines: [{"speaker": "id", "text": "..."}, ...]')
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cli", default=str(_DEFAULT_CLI))
    parser.add_argument("--tts-model", default=str(_DEFAULT_TTS_MODEL))
    parser.add_argument("--backend", default=_DEFAULT_BACKEND)
    args = parser.parse_args()

    cli = Path(args.cli)
    tts_model = Path(args.tts_model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    require(cli, f"audiocpp_cli not found at {cli} - build it first (see README Quickstart)")
    require(
        tts_model,
        f"chatterbox model not found at {tts_model} - run tools/model_manager_v2.py install chatterbox_q8_0",
    )

    voices = json.loads(Path(args.voices).read_text())
    lines = json.loads(Path(args.script).read_text())

    for speaker, ref in voices.items():
        if not Path(ref).exists():
            sys.exit(f"voice {speaker!r} references missing file: {ref}")

    line_paths = []
    manifest = []
    for i, line in enumerate(lines):
        speaker = line["speaker"]
        text = line["text"]
        if speaker not in voices:
            sys.exit(f"line {i}: unknown speaker {speaker!r}, not in --voices registry")

        out_path = out_dir / f"line_{i:04d}_{speaker}.wav"
        print(f"[{i}] {speaker}: {text!r}", file=sys.stderr)
        synthesize_with_pauses(
            text,
            out_path,
            lambda chunk, path, _ref=voices[speaker]: synthesize(cli, tts_model, args.backend, chunk, _ref, path),
        )
        line_paths.append(out_path)
        manifest.append({"line": i, "speaker": speaker, "text": text, "audio_out": str(out_path)})

    full_path = out_dir / "full.wav"
    concat_wavs(line_paths, full_path)

    manifest_path = out_dir / "narration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"lines": manifest, "full_audio": str(full_path)}))


if __name__ == "__main__":
    main()
