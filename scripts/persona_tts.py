#!/usr/bin/env python3
"""Synthesize voice-cloned speech via audio.cpp's Chatterbox family.

Thin wrapper around `audiocpp_cli --task clon --family chatterbox`, exposing
a --sentences-file/--speaker-wav interface matching a conventional
XTTS-style synthesis CLI, so it's a drop-in swap for one.

A sentence may include inline "[pause 0.6s]" markers - each spoken chunk
around a marker is synthesized separately and real silence is spliced in
for the gap, since Chatterbox has no pause token of its own and would
otherwise read the marker aloud literally.

Prints a single JSON line to stdout on success: {"wav_paths": [...]}
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from persona_turn import _DEFAULT_BACKEND, _DEFAULT_CLI, synthesize_with_pauses  # noqa: E402

_AUDIOCPP_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "audio.cpp"
_DEFAULT_MODEL = _AUDIOCPP_ROOT / "models" / "Chatterbox-GGUF" / "chatterbox-q8_0.gguf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentences-file", required=True, help="JSON file containing a list of sentence strings")
    parser.add_argument("--speaker-wav", required=True, help="Reference clip to clone the voice from")
    parser.add_argument("--language", default="en")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cli", default=str(_DEFAULT_CLI))
    parser.add_argument("--model", default=str(_DEFAULT_MODEL))
    parser.add_argument("--backend", default=_DEFAULT_BACKEND)
    args = parser.parse_args()

    cli = Path(args.cli)
    model = Path(args.model)
    if not cli.exists():
        sys.exit(f"audiocpp_cli not found at {cli} - build it first (see README Quickstart)")
    if not model.exists():
        sys.exit(f"chatterbox model not found at {model} - run tools/model_manager_v2.py install chatterbox_q8_0")

    sentences = json.loads(Path(args.sentences_file).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_paths = [out_dir / f"sentence_{i:04d}.wav" for i in range(len(sentences))]
    missing = [i for i, p in enumerate(out_paths) if not p.exists()]

    skipped = len(sentences) - len(missing)
    if skipped:
        print(f"resuming: {skipped} sentence(s) already synthesized, {len(missing)} remaining", file=sys.stderr)

    for i in missing:
        print(f"[{i + 1}/{len(sentences)}] {sentences[i]!r}", file=sys.stderr)

        def _synth_chunk(text: str, out_path: Path, _i: int = i) -> None:
            result = subprocess.run(
                [
                    str(cli),
                    "--task",
                    "clon",
                    "--family",
                    "chatterbox",
                    "--model",
                    str(model),
                    "--backend",
                    args.backend,
                    "--device",
                    "0",
                    "--language",
                    args.language,
                    "--text",
                    text,
                    "--voice-ref",
                    args.speaker_wav,
                    "--out",
                    str(out_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                sys.exit(f"audiocpp_cli failed on sentence {_i}:\n{result.stderr[-4000:]}")

        synthesize_with_pauses(sentences[i], out_paths[i], _synth_chunk)

    print(json.dumps({"wav_paths": [str(p) for p in out_paths]}))


if __name__ == "__main__":
    main()
