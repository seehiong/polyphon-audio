#!/usr/bin/env python3
"""polyphon-ai's named, diarized transcript -> LLM -> spoken reply.

    meeting.json (polyphon-ai: {speaker_name, text} segments, time-ordered)
        -> formatted "Speaker: text" transcript
        -> local LLM (given the transcript + an instruction)
        -> reply text, addressing named speakers where relevant
        -> Chatterbox voice clone -> spoken reply

The integration diagram in README's "Optional integration" section made
concrete. Not yet done: per-speaker voice selection for the *input* side
(this only speaks the persona's own single reply, it doesn't re-voice
each meeting participant) and any live/streaming connection to
polyphon-ai - this reads one already-finished transcript file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from persona_turn import (  # noqa: E402
    _DEFAULT_BACKEND,
    _DEFAULT_CLI,
    _DEFAULT_TTS_MODEL,
    query_llm,
    require,
    synthesize,
)

_DEFAULT_INSTRUCTION = (
    "Based on this meeting transcript, give a short (2-4 sentence) spoken "
    "summary of what was decided and what happens next. Address specific "
    "people by name where it's natural to do so."
)


def format_transcript(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        speaker = seg.get("speaker_name") or seg.get("speaker_id") or "Unknown"
        lines.append(f"{speaker}: {seg['text']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transcript", required=True, help="polyphon-ai transcript JSON (has speakers/segments)")
    parser.add_argument(
        "--instruction", default=_DEFAULT_INSTRUCTION, help="What to ask the LLM to do with the transcript"
    )
    parser.add_argument("--system", default=None, help="Optional system prompt (persona instructions)")
    parser.add_argument("--llm-endpoint", default="http://127.0.0.1:8082/v1")
    parser.add_argument("--llm-model", default="qwen3-30b-a3b")
    parser.add_argument(
        "--voice-ref", default=str(Path(__file__).resolve().parents[1] / "examples" / "voice_synthetic.wav")
    )
    parser.add_argument("--out", required=True, help="Output WAV path for the spoken reply")
    parser.add_argument("--cli", default=str(_DEFAULT_CLI))
    parser.add_argument("--tts-model", default=str(_DEFAULT_TTS_MODEL))
    parser.add_argument("--backend", default=_DEFAULT_BACKEND)
    args = parser.parse_args()

    cli = Path(args.cli)
    tts_model = Path(args.tts_model)
    require(cli, f"audiocpp_cli not found at {cli} - build it first (see README Quickstart)")
    require(
        tts_model,
        f"chatterbox model not found at {tts_model} - run tools/model_manager_v2.py install chatterbox_q8_0",
    )

    data = json.loads(Path(args.transcript).read_text())
    speakers = [s["name"] for s in data.get("speakers", [])]
    transcript_text = format_transcript(data["segments"])
    print(f"loaded transcript: {len(data['segments'])} segments, speakers: {speakers}", file=sys.stderr)

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append(
        {
            "role": "user",
            "content": f"Meeting transcript:\n\n{transcript_text}\n\n---\n\n{args.instruction}",
        }
    )

    print("-> querying LLM with full transcript...", file=sys.stderr)
    reply = query_llm(args.llm_endpoint, args.llm_model, messages)
    print(f"<- {reply!r}", file=sys.stderr)

    synthesize(cli, tts_model, args.backend, reply, args.voice_ref, Path(args.out))
    print(json.dumps({"speakers": speakers, "reply_text": reply, "audio_out": args.out}))


if __name__ == "__main__":
    main()
