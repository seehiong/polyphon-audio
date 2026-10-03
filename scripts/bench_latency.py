#!/usr/bin/env python3
"""Latency and VRAM benchmark for the resident audiocpp_server stack.

Three phases, each optional (--phases):

  models   Per-model cold vs warm latency and VRAM. Unloads everything, then
           touches each model in turn - first request (cold: load + run),
           then --warm-runs more (warm) - without unloading in between, so
           the VRAM column after the last model is the steady state
           server.json's `max_loaded_models: 0` gives once every model has
           been used.
  turn     One Live Chat turn as bot_fastrtc.py runs it: ASR -> streamed LLM
           reply -> TTS of the first sentence while the LLM keeps streaming.
           Reports time-to-first-audio and how much slower TTS gets while
           sharing the GPU with a streaming LLM than on an idle one.
  bargein  A long TTS request is in flight when the user "interrupts": ASR on
           the new utterance starts --barge-delay seconds in. Compares that
           ASR latency against ASR on an idle server - audio.cpp can't cancel
           a running GPU call, so the abandoned TTS keeps competing for it.

VRAM is whole-device usage (nvidia-smi or rocm-smi), reported as a delta
from the reading taken after unloading every model, so the LLM and anything
else on the GPU cancel out - keep the rest of the machine steady while it runs.
Stdlib only, like the other batch scripts; needs audiocpp_server running,
plus the LLM server for the turn phase.

Run: python scripts/bench_latency.py --llm-model qwen3-30b-a3b
Results print as tables and are saved to outputs/bench/<timestamp>.json.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from datetime import datetime
from io import BytesIO
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
# Same split bot_fastrtc.py streams sentences to TTS on.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

_TTS_TEXT = "The quick brown fox jumps over the lazy dog, then settles down for a well earned nap."
_LONG_TTS_TEXT = (
    "Once upon a time, in a village at the edge of a great forest, there lived an old clockmaker "
    "who could fix anything that ticked, and many things that did not."
)
_TURN_PROMPT = "In three short sentences, why do cats sleep so much?"

ALL_MODELS = ["nemotron-asr", "chatterbox", "voice-design", "sound-music", "sound-sfx"]


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile; values need not be sorted."""
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def summarize(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "n": len(values),
        "median_ms": round(statistics.median(values)),
        "min_ms": round(min(values)),
        "p90_ms": round(percentile(values, 90)),
    }


def first_sentence(buffer: str) -> tuple[str | None, str]:
    """(first complete sentence, remainder) once buffer holds one, else (None, buffer)."""
    parts = _SENTENCE_END_RE.split(buffer, maxsplit=1)
    if len(parts) == 2 and parts[0].strip():
        return parts[0].replace("\n", " ").strip(), parts[1]
    return None, buffer


class Gpu:
    """Whole-device VRAM readings via nvidia-smi or rocm-smi, in MiB.

    On a unified-memory AMD APU (e.g. Strix Halo/gfx1151), rocm-smi's VRAM
    counter tracks a small fixed carve-out, not the much larger system-RAM
    pool HIP actually allocates models from (audiocpp_server logs a real
    "VRAM: N MiB" figure for that pool at startup) - the models phase's
    resident_delta_mib/cumulative_mib/peak_cumulative_mib columns stay near
    zero on that hardware regardless of what actually loaded. Cold/warm
    latency numbers are unaffected and still meaningful.
    """

    def __init__(self, device: int) -> None:
        self.device = device
        self.tool = next((t for t in ("nvidia-smi", "rocm-smi") if shutil.which(t)), None)
        self.name = self._name()

    def _run(self, args: list[str]) -> str:
        return subprocess.run([self.tool, *args], capture_output=True, text=True, timeout=10, check=False).stdout

    def _name(self) -> str:
        if self.tool == "nvidia-smi":
            return self._run(["--query-gpu=name,memory.total", "--format=csv,noheader", "-i", str(self.device)]).strip()
        if self.tool == "rocm-smi":
            info = json.loads(self._run(["--showproductname", "--json"]) or "{}").get(f"card{self.device}", {})
            return info.get("Card Series") or info.get("Card series") or "AMD GPU"
        return "unknown (no nvidia-smi/rocm-smi on PATH)"

    def used_mib(self) -> float | None:
        if self.tool == "nvidia-smi":
            out = self._run(["--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(self.device)])
            return float(out.strip())
        if self.tool == "rocm-smi":
            info = json.loads(self._run(["--showmeminfo", "vram", "--json"]) or "{}").get(f"card{self.device}", {})
            used = info.get("VRAM Total Used Memory (B)")
            return int(used) / 2**20 if used is not None else None
        return None


class PeakSampler:
    """Polls Gpu.used_mib() in the background; .peak is the highest reading seen."""

    def __init__(self, gpu: Gpu, interval: float = 0.1) -> None:
        self.gpu, self.interval = gpu, interval
        self.peak: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            used = self.gpu.used_mib()
            if used is not None and (self.peak is None or used > self.peak):
                self.peak = used
            self._stop.wait(self.interval)

    def __enter__(self) -> PeakSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


def post_json(url: str, payload: dict | None, timeout: float) -> tuple[float, dict | bytes]:
    """POST, returning (elapsed ms, parsed JSON or raw bytes). Raises on HTTP errors."""
    data = json.dumps(payload).encode() if payload is not None else b""
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            content_type = resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from {url}: {e.read()[:300].decode(errors='replace')}") from e
    elapsed = (time.perf_counter() - start) * 1000
    return elapsed, json.loads(body) if "json" in content_type else body


def model_request(model: str, args: argparse.Namespace, text: str = _TTS_TEXT) -> tuple[str, dict]:
    """(endpoint path, payload) for one representative request to a server.json model id."""
    if model == "nemotron-asr":
        return "/v1/audio/transcriptions", {"model": model, "audio": str(Path(args.asr_audio).resolve())}
    if model == "chatterbox":
        return "/v1/audio/speech", {
            "model": model,
            "input": text,
            "voice_ref": {"type": "path", "path": str(Path(args.voice_ref).resolve())},
        }
    if model == "voice-design":
        return "/v1/audio/speech", {
            "model": model,
            "input": text,
            "instructions": "A calm, warm, middle-aged narrator voice with clear articulation.",
        }
    if model in ("sound-music", "sound-sfx"):
        prompt = "gentle lo-fi piano loop" if model == "sound-music" else "a wooden door creaking open"
        return "/v1/tasks/run", {"model": model, "text": prompt, "duration_seconds": args.sound_seconds}
    raise ValueError(f"no benchmark request defined for model {model!r}")


def wav_seconds(body: dict | bytes) -> float | None:
    """Duration of a WAV response body, None for non-audio (e.g. ASR JSON)."""
    if not isinstance(body, bytes) or not body.startswith(b"RIFF"):
        return None
    with wave.open(BytesIO(body), "rb") as w:
        return w.getnframes() / w.getframerate()


def call_model(model: str, args: argparse.Namespace, text: str = _TTS_TEXT, info: dict | None = None) -> float:
    """Elapsed ms for one request; fills info["audio_s"] when given and the reply is audio."""
    path, payload = model_request(model, args, text)
    elapsed, body = post_json(args.audiocpp_url + path, payload, timeout=args.timeout)
    if info is not None:
        info["audio_s"] = wav_seconds(body)
    return elapsed


def unload_all(args: argparse.Namespace) -> None:
    post_json(args.audiocpp_url + "/v1/tasks/unload_all_models", None, timeout=args.timeout)
    time.sleep(1.0)  # let the driver release memory before the next reading


def phase_models(args: argparse.Namespace, gpu: Gpu) -> dict:
    print("\n== models: cold vs warm, cumulative VRAM ==")
    unload_all(args)
    baseline = gpu.used_mib()
    rows = []
    for model in args.models:
        before = gpu.used_mib()
        row: dict = {"model": model}
        try:
            with PeakSampler(gpu) as sampler:
                row["cold_ms"] = round(call_model(model, args))
            time.sleep(0.5)
            after = gpu.used_mib()
            warm = [call_model(model, args) for _ in range(args.warm_runs)]
            row["warm"] = summarize(warm)
            if before is not None and after is not None:
                row["resident_delta_mib"] = round(after - before)
                row["cumulative_mib"] = round(after - baseline)
                row["peak_cumulative_mib"] = round(sampler.peak - baseline) if sampler.peak is not None else None
        except Exception as e:  # noqa: BLE001 - one model failing (e.g. OOM) shouldn't end the run
            row["error"] = str(e)
        rows.append(row)
        warm = row.get("warm", {})
        print(
            f"  {model:14s} cold {row.get('cold_ms', '-'):>7} ms   warm median {warm.get('median_ms', '-'):>6} ms"
            f"   +{row.get('resident_delta_mib', '?')} MiB (total {row.get('cumulative_mib', '?')},"
            f" peak {row.get('peak_cumulative_mib', '?')})"
            + (f"   ERROR {row['error'][:120]}" if "error" in row else "")
        )
    return {"baseline_mib": baseline, "rows": rows}


def stream_llm(args: argparse.Namespace, on_first_sentence) -> dict:
    """Streams one reply; calls on_first_sentence(text) the moment one completes."""
    payload = {
        "model": args.llm_model,
        "messages": [{"role": "user", "content": _TURN_PROMPT}],
        "temperature": 0.7,
        "stream": True,
        "max_tokens": 300,
        "n_predict": 300,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{args.llm_endpoint}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    ttft = first_sentence_ms = None
    buffer, fired = "", False
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line.removeprefix("data: "))["choices"][0]["delta"].get("content") or ""
            if not delta:
                continue
            if ttft is None:
                ttft = (time.perf_counter() - start) * 1000
            buffer += delta
            if not fired:
                sentence, buffer = first_sentence(buffer)
                if sentence:
                    fired = True
                    first_sentence_ms = (time.perf_counter() - start) * 1000
                    on_first_sentence(sentence)
    total = (time.perf_counter() - start) * 1000
    if not fired and buffer.strip():  # a reply with no sentence break at all
        first_sentence_ms = total
        on_first_sentence(buffer.strip())
    return {"ttft_ms": ttft, "first_sentence_ms": first_sentence_ms, "total_ms": total}


def phase_turn(args: argparse.Namespace) -> dict:
    print("\n== turn: ASR -> LLM stream -> first-sentence TTS ==")
    # Warm everything first: this phase is about steady-state turns.
    call_model("nemotron-asr", args)
    call_model("chatterbox", args)
    stream_llm(args, lambda _sentence: None)
    idle_tts = [call_model("chatterbox", args) for _ in range(args.warm_runs)]
    runs = []
    for i in range(args.turn_runs):
        asr_ms = call_model("nemotron-asr", args)
        tts_result: dict = {}

        def tts_first(sentence: str, _result=tts_result) -> None:
            def synth() -> None:
                _result["sentence"] = sentence
                _result["ms"] = call_model("chatterbox", args, text=sentence, info=_result)

            _result["thread"] = threading.Thread(target=synth)
            _result["thread"].start()

        llm = stream_llm(args, tts_first)
        if "thread" in tts_result:
            tts_result["thread"].join()
        run = {
            "asr_ms": round(asr_ms),
            "llm_ttft_ms": round(llm["ttft_ms"] or 0),
            "llm_first_sentence_ms": round(llm["first_sentence_ms"] or 0),
            "llm_total_ms": round(llm["total_ms"]),
            "tts_first_sentence_ms": round(tts_result.get("ms", 0)),
            "first_sentence": tts_result.get("sentence"),
            "first_sentence_audio_s": round(tts_result.get("audio_s") or 0, 2),
        }
        # End of user speech -> first reply audio ready (VAD pause window not included).
        run["time_to_first_audio_ms"] = run["asr_ms"] + run["llm_first_sentence_ms"] + run["tts_first_sentence_ms"]
        runs.append(run)
        print(
            f"  run {i + 1}: ASR {run['asr_ms']} + LLM first sentence {run['llm_first_sentence_ms']}"
            f" (ttft {run['llm_ttft_ms']}) + TTS {run['tts_first_sentence_ms']}"
            f" = {run['time_to_first_audio_ms']} ms to first audio"
            f"  [{run['first_sentence_audio_s']}s audio: {run['first_sentence']!r}]"
        )
    idle = summarize(idle_tts)
    print(f"  TTS on an idle GPU (same voice, fixed sentence): median {idle.get('median_ms')} ms")
    return {
        "runs": runs,
        "time_to_first_audio": summarize([r["time_to_first_audio_ms"] for r in runs]),
        "tts_during_llm_stream": summarize([r["tts_first_sentence_ms"] for r in runs]),
        "tts_idle_gpu": idle,
    }


def phase_bargein(args: argparse.Namespace) -> dict:
    print("\n== bargein: ASR while an abandoned TTS request is still running ==")
    call_model("nemotron-asr", args)
    call_model("chatterbox", args)
    idle_asr = [call_model("nemotron-asr", args) for _ in range(args.warm_runs)]
    contended = []
    for _ in range(args.warm_runs):
        tts = threading.Thread(target=call_model, args=("chatterbox", args, _LONG_TTS_TEXT))
        tts.start()
        time.sleep(args.barge_delay)
        contended.append(call_model("nemotron-asr", args))
        tts.join()
    result = {"asr_idle": summarize(idle_asr), "asr_during_tts": summarize(contended)}
    print(
        f"  ASR idle median {result['asr_idle']['median_ms']} ms,"
        f" during in-flight TTS median {result['asr_during_tts']['median_ms']} ms"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audiocpp-url", default="http://127.0.0.1:8091")
    parser.add_argument("--llm-endpoint", default="http://127.0.0.1:8082/v1")
    parser.add_argument("--llm-model", default="qwen3-30b-a3b")
    phase_names = ["models", "turn", "bargein"]
    parser.add_argument("--phases", nargs="+", default=phase_names, choices=phase_names)
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, help="server.json model ids for the models phase")
    parser.add_argument("--warm-runs", type=int, default=5)
    parser.add_argument("--turn-runs", type=int, default=5)
    parser.add_argument(
        "--barge-delay", type=float, default=0.3, help="seconds into the TTS request the user interrupts"
    )
    parser.add_argument("--sound-seconds", type=float, default=5.0, help="clip length for the stable_audio models")
    parser.add_argument("--voice-ref", default=str(_EXAMPLES_DIR / "voice_synthetic.wav"))
    parser.add_argument("--asr-audio", default=str(_EXAMPLES_DIR / "voice.wav"))
    parser.add_argument("--device", type=int, default=0, help="GPU index for VRAM readings")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--label", default="", help="free-text note saved with the results, e.g. the backend")
    parser.add_argument("--out-dir", default=str(_REPO_ROOT / "outputs" / "bench"))
    args = parser.parse_args()

    gpu = Gpu(args.device)
    print(f"GPU: {gpu.name}  |  audio.cpp: {args.audiocpp_url}  |  LLM: {args.llm_model} @ {args.llm_endpoint}")
    results: dict = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "label": args.label,
        "platform": platform.platform(),
        "gpu": gpu.name,
        "audiocpp_url": args.audiocpp_url,
        "llm_model": args.llm_model,
        "settings": {k: getattr(args, k) for k in ("warm_runs", "turn_runs", "barge_delay", "sound_seconds")},
    }
    phases = {
        "models": lambda: phase_models(args, gpu),
        "turn": lambda: phase_turn(args),
        "bargein": lambda: phase_bargein(args),
    }
    for name in args.phases:
        try:
            results[name] = phases[name]()
        except Exception as e:  # noqa: BLE001 - keep whatever the other phases measured
            results[name] = {"error": str(e)}
            print(f"  {name} phase failed: {e}", file=sys.stderr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bench_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
