#!/usr/bin/env python3
"""Multi-turn conversation, entirely local: same loop as persona_turn.py,
but the LLM keeps message history across turns instead of treating each
prompt as a fresh conversation.

    turn 1 prompt -> LLM (system + [turn1])               -> reply 1 -> Chatterbox -> turn_0000.wav
    turn 2 prompt -> LLM (system + [turn1, reply1, turn2]) -> reply 2 -> Chatterbox -> turn_0001.wav
    ...

Each turn's prompt is text or a WAV (transcribed via Nemotron ASR, same as
persona_turn.py's --prompt-audio). Takes a --script JSON file: a list of
turns, each `{"text": "..."}` or `{"audio": "path/to/clip.wav"}`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from persona_turn import (  # noqa: E402
    _DEFAULT_ASR_MODEL,
    _DEFAULT_BACKEND,
    _DEFAULT_CLI,
    _DEFAULT_TTS_MODEL,
    query_llm,
    require,
    synthesize,
    transcribe_audio,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--script", required=True, help='JSON list of turns: [{"text": "..."}, {"audio": "..."}]')
    parser.add_argument("--system", default=None, help="Optional system prompt (persona instructions)")
    parser.add_argument("--llm-endpoint", default="http://127.0.0.1:8082/v1")
    parser.add_argument("--llm-model", default="qwen3-30b-a3b")
    parser.add_argument("--voice-ref", default=str(Path(__file__).resolve().parents[1] / "examples" / "voice.wav"))
    parser.add_argument(
        "--voices", default=None, help="Named voice registry JSON (same format as narrate.py); overrides --voice-ref"
    )
    parser.add_argument("--voice", default=None, help="Voice id to look up in --voices")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cli", default=str(_DEFAULT_CLI))
    parser.add_argument("--tts-model", default=str(_DEFAULT_TTS_MODEL))
    parser.add_argument("--asr-model", default=str(_DEFAULT_ASR_MODEL))
    parser.add_argument("--backend", default=_DEFAULT_BACKEND)
    args = parser.parse_args()

    cli = Path(args.cli)
    tts_model = Path(args.tts_model)
    asr_model = Path(args.asr_model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    require(cli, f"audiocpp_cli not found at {cli} - build it first (see README Quickstart)")
    require(
        tts_model,
        f"chatterbox model not found at {tts_model} - run tools/model_manager_v2.py install chatterbox_q8_0",
    )

    if args.voices:
        voices = json.loads(Path(args.voices).read_text())
        if not args.voice or args.voice not in voices:
            sys.exit(f"--voice must be one of {list(voices)} when --voices is given")
        voice_ref = voices[args.voice]
    else:
        voice_ref = args.voice_ref

    turns = json.loads(Path(args.script).read_text())
    if any("audio" in turn for turn in turns):
        require(
            asr_model,
            f"nemotron_asr model not found at {asr_model} - run "
            "tools/model_manager_v2.py install nemotron_asr_q8_0",
        )
    messages = [{"role": "system", "content": args.system}] if args.system else []
    transcript = []

    for i, turn in enumerate(turns):
        if "audio" in turn:
            print(f"[{i}] transcribing {turn['audio']!r}...", file=sys.stderr)
            prompt = transcribe_audio(cli, asr_model, args.backend, turn["audio"])
        else:
            prompt = turn["text"]
        print(f"[{i}] user: {prompt!r}", file=sys.stderr)

        messages.append({"role": "user", "content": prompt})
        reply = query_llm(args.llm_endpoint, args.llm_model, messages)
        print(f"[{i}] assistant: {reply!r}", file=sys.stderr)
        messages.append({"role": "assistant", "content": reply})

        out_path = out_dir / f"turn_{i:04d}.wav"
        synthesize(cli, tts_model, args.backend, reply, voice_ref, out_path)
        transcript.append({"turn": i, "prompt": prompt, "reply": reply, "audio_out": str(out_path)})

    transcript_path = out_dir / "session_transcript.json"
    transcript_path.write_text(json.dumps(transcript, indent=2))
    print(json.dumps({"turns": transcript, "transcript_file": str(transcript_path)}))


if __name__ == "__main__":
    main()
