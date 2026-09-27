#!/usr/bin/env python3
"""One conversational turn, entirely local:

    prompt (text, or spoken audio via --prompt-audio)
        -> [Nemotron ASR, if audio]
        -> local LLM (OpenAI-compatible /chat/completions) -> reply text
        -> Chatterbox voice clone (audio.cpp) -> reply audio

Not real-time and not wired to polyphon-ai's diarized transcripts - one
prompt in, one WAV out. Also home to the shared helpers (require,
synthesize, query_llm, transcribe_audio) the other scripts here import.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import wave
from collections.abc import Callable
from pathlib import Path

_AUDIOCPP_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "audio.cpp"

# (cli path, --backend value) for each build layout the setup scripts produce:
# setup_audiocpp.sh's Linux HIP build, then setup_audiocpp.bat's Windows presets.
_BUILD_CANDIDATES = [
    (_AUDIOCPP_ROOT / "build_hip" / "bin" / "audiocpp_cli", "hip"),
    *(
        (_AUDIOCPP_ROOT / "build" / f"windows-{b}-release" / "bin" / "audiocpp_cli.exe", b)
        for b in ("cuda", "vulkan", "cpu")
    ),
]


def detect_build(
    candidates: list[tuple[Path, str]] = _BUILD_CANDIDATES, windows: bool = sys.platform == "win32"
) -> tuple[Path, str]:
    """The first audio.cpp build that exists, as (audiocpp_cli path, backend).

    Falls back to this platform's first candidate when none is built yet, so
    the "not found" error points at a path that makes sense on this OS.
    """
    for cli, backend in candidates:
        if cli.exists():
            return cli, backend
    return next(((c, b) for c, b in candidates if (c.suffix == ".exe") == windows), candidates[0])


_DEFAULT_CLI, _DEFAULT_BACKEND = detect_build()
_DEFAULT_TTS_MODEL = _AUDIOCPP_ROOT / "models" / "Chatterbox-GGUF" / "chatterbox-q8_0.gguf"
_DEFAULT_ASR_MODEL = (
    _AUDIOCPP_ROOT / "models" / "Nemotron-3.5-ASR-Streaming-0.6B-GGUF" / "nemotron-3.5-asr-streaming-0.6b-q8_0.gguf"
)


def require(path: Path, message: str) -> None:
    if not path.exists():
        sys.exit(message)


def transcribe_audio(cli: Path, model_path: Path, backend: str, audio_path: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        text_out = Path(tmp.name)
    try:
        result = subprocess.run(
            [
                str(cli),
                "--task", "asr",
                "--family", "nemotron_asr",
                "--model", str(model_path),
                "--backend", backend,
                "--device", "0",
                "--audio", audio_path,
                "--text-out", str(text_out),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            sys.exit(f"audiocpp_cli ASR failed:\n{result.stderr[-4000:]}")
        return text_out.read_text().strip()
    finally:
        text_out.unlink(missing_ok=True)


def query_llm(endpoint: str, model: str, messages: list[dict]) -> str:
    body = json.dumps({"model": model, "messages": messages, "temperature": 0.7}).encode()
    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read())
    return payload["choices"][0]["message"]["content"].strip()


_PAUSE_RE = re.compile(r"\[pause[:\s]+(\d+(?:\.\d+)?)\s*s?\]", re.IGNORECASE)


def parse_pause_segments(text: str) -> list[str | float]:
    """Splits text on inline "[pause Ns]" markers into an ordered list of
    spoken chunks (str) and silence gaps in seconds (float). None of
    audio.cpp's TTS families understand a pause/break token, so a marker
    left in the text gets read aloud literally - callers synthesize each
    chunk on its own and splice real silence in for the float entries.
    """
    items: list[str | float] = []
    pos = 0
    for m in _PAUSE_RE.finditer(text):
        chunk = text[pos : m.start()].strip()
        if chunk:
            items.append(chunk)
        items.append(float(m.group(1)))
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        items.append(tail)
    return items


def silent_wav_frames(seconds: float, params: wave._wave_params) -> bytes:
    n_frames = max(0, round(seconds * params.framerate))
    return b"\x00" * (n_frames * params.nchannels * params.sampwidth)


def synthesize_with_pauses(text: str, out_path: Path, synth_chunk: Callable[[str, Path], None]) -> None:
    """Splits text on "[pause Ns]" markers, calls synth_chunk(chunk_text,
    tmp_wav_path) to render each spoken chunk, and stitches the results
    into out_path with real silence spliced in for the pause gaps. Falls
    through to a single synth_chunk call when there are no markers.
    """
    items = parse_pause_segments(text)
    if not any(isinstance(it, float) for it in items):
        synth_chunk(text.strip(), out_path)
        return

    tmp_dir = out_path.parent / f".{out_path.stem}_segments"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    resolved: list[bytes | float] = []
    params = None
    seg_i = 0
    try:
        for item in items:
            if not isinstance(item, str):
                resolved.append(float(item))
                continue
            seg_path = tmp_dir / f"seg_{seg_i:03d}.wav"
            seg_i += 1
            synth_chunk(item, seg_path)
            with wave.open(str(seg_path), "rb") as f:
                if params is None:
                    params = f.getparams()
                resolved.append(f.readframes(f.getnframes()))
        if params is None:
            sys.exit(f"{out_path}: text is only pause markers, nothing to synthesize")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_path), "wb") as out:
            out.setparams(params)
            for item in resolved:
                out.writeframes(item if isinstance(item, bytes) else silent_wav_frames(item, params))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def synthesize(cli: Path, model_path: Path, backend: str, text: str, voice_ref: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            str(cli),
            "--task", "clon",
            "--family", "chatterbox",
            "--model", str(model_path),
            "--backend", backend,
            "--device", "0",
            "--text", text,
            "--voice-ref", voice_ref,
            "--out", str(out_path),
            "--metrics",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.exit(f"audiocpp_cli TTS failed:\n{result.stderr[-4000:]}")
    print(result.stdout, file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    prompt_source = parser.add_mutually_exclusive_group(required=True)
    prompt_source.add_argument("--prompt", help="What to say to the LLM, as text")
    prompt_source.add_argument(
        "--prompt-audio", help="What to say to the LLM, as a WAV file (transcribed via Nemotron ASR)"
    )
    parser.add_argument("--system", default=None, help="Optional system prompt (persona instructions)")
    parser.add_argument("--llm-endpoint", default="http://127.0.0.1:8082/v1")
    parser.add_argument("--llm-model", default="qwen3-30b-a3b")
    parser.add_argument("--voice-ref", default=str(Path(__file__).resolve().parents[1] / "examples" / "voice.wav"))
    parser.add_argument("--out", required=True, help="Output WAV path for the spoken reply")
    parser.add_argument("--cli", default=str(_DEFAULT_CLI))
    parser.add_argument("--tts-model", default=str(_DEFAULT_TTS_MODEL))
    parser.add_argument("--asr-model", default=str(_DEFAULT_ASR_MODEL))
    parser.add_argument("--backend", default=_DEFAULT_BACKEND)
    args = parser.parse_args()

    cli = Path(args.cli)
    tts_model = Path(args.tts_model)
    asr_model = Path(args.asr_model)
    require(cli, f"audiocpp_cli not found at {cli} - build it first (see README Quickstart)")
    require(
        tts_model,
        f"chatterbox model not found at {tts_model} - run tools/model_manager_v2.py install chatterbox_q8_0",
    )

    if args.prompt_audio:
        require(
            asr_model,
            f"nemotron_asr model not found at {asr_model} - run "
            "tools/model_manager_v2.py install nemotron_asr_q8_0",
        )
        print(f"-> transcribing {args.prompt_audio!r}...", file=sys.stderr)
        prompt = transcribe_audio(cli, asr_model, args.backend, args.prompt_audio)
        print(f"<- heard: {prompt!r}", file=sys.stderr)
    else:
        prompt = args.prompt

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt})

    print(f"-> {args.llm_model}: {prompt!r}", file=sys.stderr)
    reply = query_llm(args.llm_endpoint, args.llm_model, messages)
    print(f"<- {reply!r}", file=sys.stderr)

    synthesize_with_pauses(
        reply,
        Path(args.out),
        lambda chunk, path: synthesize(cli, tts_model, args.backend, chunk, args.voice_ref, path),
    )
    print(json.dumps({"prompt_text": prompt, "reply_text": reply, "audio_out": args.out}))


if __name__ == "__main__":
    main()
