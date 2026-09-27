"""Real-time voice loop, FastRTC-based: browser mic/speaker -> Silero VAD
(ReplyOnPause) -> audio.cpp Nemotron ASR (resident server) -> local LLM ->
audio.cpp Chatterbox (resident server, cloned voice) -> spoken reply, with
interruption. Reused as-is by scripts/studio.py's Live Chat tab (imports
`build_live_chat_ui`/`Conversation`).

Live transcript + latency stats render via a custom gr.Blocks layout wired
directly to the WebRTC component's additional_outputs, not Stream's
auto-generated UI. Transcript is a plain gr.HTML (output-only, rendered
fresh each turn from conversation.messages), not gr.Chatbot. Replies are
streamed and spoken sentence-by-sentence rather than waiting for the full
reply, so the first sentence's audio starts well before the rest is even
generated; each sentence's transcript text is revealed right as its audio
plays. Markdown the LLM writes is stripped before TTS - it both breaks
sentence-boundary detection and gets vocalized as a garbled sound otherwise.
A sentence may include inline "[pause 0.6s]" markers - see speak()/
parse_pause_segments (persona_turn.py) for how those are honored.

Prerequisites (see README "Quickstart"):
  1. audiocpp_server --config vendor/audio.cpp/server.json, from
     vendor/audio.cpp/build_hip/bin/ (Linux AMD) or
     vendor/audio.cpp/build/windows-<backend>-release/bin/ (Windows)
  2. A local OpenAI-compatible LLM server (e.g. llama-server) on --llm-endpoint

Run: .venv-fastrtc/bin/python scripts/realtime/bot_fastrtc.py
     (Windows: .venv-fastrtc/Scripts/python.exe scripts/realtime/bot_fastrtc.py)
Then open the printed https://localhost:7861 URL in a browser.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from collections.abc import AsyncGenerator, Mapping
from datetime import datetime
from io import BytesIO
from pathlib import Path

import fastrtc
import gradio as gr
import httpx
import numpy as np
from dotenv import load_dotenv
from fastrtc import AdditionalOutputs, AlgoOptions, ReplyOnPause, WebRTC
from fastrtc.webrtc_connection_mixin import WebRTCConnectionMixin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from persona_turn import parse_pause_segments, silent_wav_frames  # noqa: E402

# Workaround for a fastrtc 0.0.34 bug (webrtc_connection_mixin.py's
# handle_offer): its ICE candidate parser requires >=10 space-separated
# tokens, but some browsers send a minimal 8-token RFC 5245 candidate,
# which it then silently drops - failing a connection with nothing but an
# "Invalid candidate format" log line. Pad with harmless extra tokens to
# satisfy the length check. Safe to remove once upstream fixes this.
_original_handle_offer = WebRTCConnectionMixin.handle_offer


async def _patched_handle_offer(self, body, set_outputs):
    msg_type = body.get("type")
    webrtc_id = body.get("webrtc_id")
    if msg_type == "ice-candidate" and "candidate" in body:
        candidate_str = body["candidate"].get("candidate", "")
        parts = candidate_str.split()
        if parts and len(parts) < 10 and parts[0].startswith("candidate:"):
            body["candidate"]["candidate"] = candidate_str + " generation 0 ufrag 0000 network-id 1"

    response = await _original_handle_offer(self, body, set_outputs)

    # One line per connection attempt (not per candidate, which trickles a
    # dozen+ per connection) plus state transitions below - enough to
    # diagnose a connection problem without spamming normal output.
    if msg_type == "offer":
        print(f"[webrtc {webrtc_id}] offer -> status={getattr(response, 'status_code', '?')} pcs_open={len(self.pcs)}")

        if webrtc_id in self.pcs:
            pc = self.pcs[webrtc_id]

            @pc.on("iceconnectionstatechange")
            def _log_ice_state(_pc=pc, _id=webrtc_id):
                print(f"[webrtc {_id}] iceConnectionState -> {_pc.iceConnectionState}")

            @pc.on("connectionstatechange")
            def _log_conn_state(_pc=pc, _id=webrtc_id):
                print(f"[webrtc {_id}] connectionState -> {_pc.connectionState}")

    return response


WebRTCConnectionMixin.handle_offer = _patched_handle_offer

# Workaround for a second, client-side fastrtc bug: the "wave" visualizer's
# amplitude animation always reads the remote (received) audio stream, not
# the local mic, so it only animates during the assistant's reply and stays
# flat while the user talks. Patches the compiled JS asset directly (not
# reachable via a Python-level monkeypatch) to prefer the local mic stream
# whenever one exists. Idempotent; safe to delete once upstream fixes it.
_webrtc_js_path = Path(fastrtc.__file__).parent / "templates" / "component" / "index-EQgvOkvr.js"
_webrtc_js_broken = 'a === "send" ? ee : K.srcObject'
_webrtc_js_fixed = "ee || K.srcObject"
try:
    _webrtc_js_text = _webrtc_js_path.read_text(encoding="utf-8")
    if _webrtc_js_broken in _webrtc_js_text:
        _webrtc_js_path.write_text(
            _webrtc_js_text.replace(_webrtc_js_broken, _webrtc_js_fixed),
            encoding="utf-8",
        )
        print(f"!! Patched {_webrtc_js_path.name}: wave visualizer now shows the mic, not just replies")
except OSError as e:
    print(f"!! Could not patch {_webrtc_js_path.name} (wave visualizer will show replies, not the mic): {e}")

# .env holds this machine's actual current server ports; DEFAULT_* below are
# just fresh-checkout fallbacks.
load_dotenv()

_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"

DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful voice assistant. The current date and time is "
    "{current_datetime} - trust this over whatever date your training data "
    "implies, and answer date/time questions directly from it instead of "
    "guessing or second-guessing the user who tells you the date. Keep replies "
    "short and conversational - this is a spoken conversation, not a chat "
    "window. Aim for 2-3 sentences per turn; if a topic genuinely needs more, "
    "give the short version first and offer to go deeper rather than "
    "delivering it all at once. Always reply in {language}, regardless of "
    "what language the input seems to be in. Do not use markdown formatting "
    "(no *asterisks*, `backticks`, or # headers) - this is spoken aloud, "
    "plain sentences only."
)
DEFAULT_LANGUAGE = "English"
LANGUAGE_CHOICES = ["English", "Mandarin Chinese", "Spanish", "French", "Japanese", "Korean", "Malay"]
DEFAULT_AUDIOCPP_URL = os.environ.get("POLYPHON_AUDIOCPP_URL", "http://127.0.0.1:8091")
DEFAULT_LLM_ENDPOINT = os.environ.get("POLYPHON_LLM_ENDPOINT", "http://127.0.0.1:8082/v1")
DEFAULT_LLM_MODEL = os.environ.get("POLYPHON_LLM_MODEL", "qwen3-30b-a3b")
DEFAULT_VOICE_REF = str(_EXAMPLES_DIR / "voice_synthetic.wav")
# Chatterbox's own default; overridden per-voice via the registry.
DEFAULT_EXAGGERATION = 0.5

# Shared with scripts/studio.py's Voice Library tab (both need the voice
# registry) rather than duplicated there.
VOICES_DIR = _EXAMPLES_DIR.parent / "voices"
VOICES_REGISTRY_PATH = VOICES_DIR / "registry.json"


def load_voice_registry() -> list[dict]:
    if not VOICES_REGISTRY_PATH.exists():
        return []
    return json.loads(VOICES_REGISTRY_PATH.read_text()).get("voices", [])


def save_voice_registry(voices: list[dict]) -> None:
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    VOICES_REGISTRY_PATH.write_text(json.dumps({"voices": voices}, indent=2))


def voice_dropdown_choices(voices: list[dict]) -> list[tuple[str, str]]:
    """(label, id) pairs - gr.Dropdown shows the label, returns the id."""
    return [(f"{v['label']} ({v['source']})", v["id"]) for v in voices]

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
# TTS has no concept of markdown - it either garbles the symbol or breaks
# sentence-boundary detection (e.g. "bright.*" isn't ". "). Deleted outright
# rather than matched as paired **bold**/*italic* spans, which would need
# handling a span split across streamed chunks.
_TTS_UNSAFE_CHARS = str.maketrans({c: None for c in "*_`#"})

# Applies only to time-to-first-audio, not total turn time.
_TTFB_GOOD_MS = 500
_TTFB_SLOW_MS = 1500

# Message counts (user+assistant, 2 per turn), not tokens - keeps
# self.messages bounded so a long session doesn't exceed the LLM's context
# window.
_COMPACT_TRIGGER_AT = 20
_COMPACT_KEEP_RECENT = 12


async def fetch_speech_wav(
    client: httpx.AsyncClient, audiocpp_url: str, text: str, voice_ref: str, exaggeration: float
) -> bytes:
    """One /v1/audio/speech call, no pause handling - raw WAV bytes back."""
    resp = await client.post(
        f"{audiocpp_url}/v1/audio/speech",
        json={
            "model": "chatterbox",
            "input": text,
            "voice_ref": {"type": "path", "path": voice_ref},
            "options": {"exaggeration": exaggeration},
        },
    )
    resp.raise_for_status()
    return resp.content


async def synthesize_speech_with_pauses(
    client: httpx.AsyncClient, audiocpp_url: str, text: str, voice_ref: str, exaggeration: float
) -> bytes:
    """Like fetch_speech_wav, but honors inline "[pause Ns]" markers by
    synthesizing each chunk around a marker separately and splicing real
    silence in for the gap - Chatterbox has no pause token of its own and
    would otherwise read the marker aloud literally. Shared by Live Chat
    (Conversation.speak) and studio.py's Text to Speech tab, so the two
    don't drift with only one of them handling pauses.
    """
    items = parse_pause_segments(text)
    if not any(isinstance(it, float) for it in items):
        return await fetch_speech_wav(client, audiocpp_url, text, voice_ref, exaggeration)

    params = None
    frames: list[bytes] = []
    for item in items:
        if isinstance(item, str):
            wav_bytes = await fetch_speech_wav(client, audiocpp_url, item, voice_ref, exaggeration)
            with wave.open(BytesIO(wav_bytes), "rb") as w:
                if params is None:
                    params = w.getparams()
                frames.append(w.readframes(w.getnframes()))
        elif params is not None:
            frames.append(silent_wav_frames(item, params))

    if params is None:
        raise ValueError(f"{text!r}: only pause markers, nothing to synthesize")

    buf = BytesIO()
    with wave.open(buf, "wb") as out:
        out.setparams(params)
        for f in frames:
            out.writeframes(f)
    return buf.getvalue()


class Conversation:
    """Wraps the three local services + running message history for one session."""

    def __init__(
        self,
        *,
        audiocpp_url: str,
        llm_endpoint: str,
        llm_model: str,
        voice_ref: str,
        system_prompt_template: str,
        language: str = DEFAULT_LANGUAGE,
        debug_dir: Path | None = None,
        exaggeration: float = DEFAULT_EXAGGERATION,
    ):
        self.audiocpp_url = audiocpp_url.rstrip("/")
        self.llm_endpoint = llm_endpoint.rstrip("/")
        self.llm_model = llm_model
        self.voice_ref = voice_ref
        # Chatterbox's own default; a flat 0.5 can read as robotic for a
        # short/flat reference clip.
        self.exaggeration = exaggeration
        self.system_prompt_template = system_prompt_template
        self.language = language
        # Rebuilt fresh from system_prompt_template + language on every
        # ask_stream() call (no system entry stored here), so a mid-session
        # language change applies immediately.
        self.messages: list[dict[str, str]] = []
        # Full, untrimmed history for the transcript UI/session_log.
        # self.messages (above) is what the LLM sees and may be compacted by
        # maybe_compact() - this split keeps the visible transcript intact
        # even when the LLM's context is folded down.
        self.display_messages: list[dict[str, str]] = []
        # Folded summary of turns older than _COMPACT_KEEP_RECENT (see
        # maybe_compact()).
        self.summary = ""
        # Async throughout so FastRTC can actually cancel an in-flight reply
        # on interruption - ReplyOnPause cancels via generator.aclose(),
        # which only cooperates with an async generator; a sync generator's
        # blocking call keeps running regardless.
        self.client = httpx.AsyncClient(timeout=60.0)
        self.turn_history: list[dict[str, float]] = []
        self.debug_dir = debug_dir
        self.session_log: list[dict] = []
        self.session_log_path = debug_dir / "session_log.json" if debug_dir else None
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def record_turn(self, *, prompt: str, reply: str, **metrics: float) -> None:
        self.turn_history.append(metrics)
        self.session_log.append(
            {"timestamp": time.time(), "prompt": prompt, "reply": reply, "metrics_s": metrics}
        )
        if self.session_log_path:
            # Rewritten whole each turn, not appended - sessions here are
            # short, and this keeps the file always-valid JSON.
            self.session_log_path.write_text(json.dumps(self.session_log, indent=2))

    def add_user_turn(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self.display_messages.append({"role": "user", "content": text})

    async def maybe_compact(self) -> None:
        """Folds turns older than _COMPACT_KEEP_RECENT into self.summary via
        one extra LLM call, so self.messages doesn't grow unbounded. Called
        after a turn's reply is already spoken, so it never adds latency to
        the turn the user is waiting on. Failures are caught and skipped -
        keep working with an uncompacted, growing history rather than break
        the session over a failed summarization call.
        """
        if len(self.messages) <= _COMPACT_TRIGGER_AT:
            return
        to_summarize, self.messages = self.messages[:-_COMPACT_KEEP_RECENT], self.messages[-_COMPACT_KEEP_RECENT:]
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in to_summarize)
        prior = f"Earlier summary: {self.summary}\n\n" if self.summary else ""
        prompt = (
            f"{prior}Summarize the conversation below concisely, preserving names, "
            f"facts, preferences, and decisions mentioned. A few sentences, plain "
            f"prose, no preamble.\n\n{transcript}"
        )
        try:
            resp = await self.client.post(
                f"{self.llm_endpoint}/chat/completions",
                json={
                    "model": self.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "n_predict": 200,
                    "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"},
                },
                timeout=20.0,
            )
            resp.raise_for_status()
            self.summary = resp.json()["choices"][0]["message"]["content"].strip()
            print(f"[compact] folded {len(to_summarize)} messages into summary, {len(self.messages)} kept verbatim")
        except httpx.HTTPError as e:
            # Put the messages back and retry next turn rather than losing them.
            self.messages = to_summarize + self.messages
            print(f"!! compaction failed, will retry next turn: {e!r}")

    def set_language(self, language: str) -> None:
        self.language = language

    def set_voice_ref(self, voice_ref: str, exaggeration: float = DEFAULT_EXAGGERATION) -> None:
        self.voice_ref = voice_ref
        self.exaggeration = exaggeration

    async def check_readiness(self) -> str:
        """Pings both backing services - a dead mic/WebRTC connection and a
        down LLM or audiocpp_server look identical otherwise (no reply,
        nothing in the UI to tell them apart).
        """

        async def ping(label: str, url: str) -> str:
            try:
                resp = await self.client.get(url, timeout=3.0)
                resp.raise_for_status()
                return f"\U0001f7e2 {label}"
            except httpx.HTTPError as e:
                return f"\U0001f534 {label} ({e.__class__.__name__})"

        audiocpp_status, llm_status = await asyncio.gather(
            ping("audio.cpp", f"{self.audiocpp_url}/v1/models"),
            ping(f"LLM ({self.llm_model})", f"{self.llm_endpoint}/models"),
        )
        return f"{audiocpp_status}&nbsp;&nbsp;&nbsp;{llm_status}"

    def transcript_html(self, *, spoken_this_turn: str = "") -> str:
        """HTML render of self.display_messages, the full untrimmed history
        (distinct from self.messages, which maybe_compact() may fold down
        for the LLM). Output-only, never round-tripped through
        additional_inputs.

        `spoken_this_turn` is this turn's sentences spoken so far, not yet
        committed to display_messages - revealed one sentence at a time,
        right as each one's audio is about to play.
        """
        parts = []
        for m in self.display_messages:
            speaker = "You" if m["role"] == "user" else "Assistant"
            parts.append(f"<p><b>{speaker}:</b> {html.escape(m['content'])}</p>")
        if spoken_this_turn:
            parts.append(f"<p><b>Assistant:</b> {html.escape(spoken_this_turn)}</p>")
        return "\n".join(parts)

    def stats_report(self, live: Mapping[str, float | None]) -> str:
        def fmt_ms(v: float | None) -> str:
            return f"{v * 1000:.0f}ms" if v is not None else "-"

        def label(key: str) -> str:
            return {"asr": "ASR", "ttfb": "Time to first audio (TTFB)"}.get(key, key.replace("_", " ").title())

        def quality(key: str, v: float | None) -> str:
            if key != "ttfb" or v is None:
                return ""
            ms = v * 1000
            if ms < _TTFB_GOOD_MS:
                return "  [good]"
            if ms < _TTFB_SLOW_MS:
                return "  [ok]"
            return "  [slow]"

        lines = ["THIS TURN", "-" * 28]
        lines += [f"{label(k):<26} {fmt_ms(v)}{quality(k, v)}" for k, v in live.items()]

        if self.turn_history:
            keys = sorted({k for turn in self.turn_history for k in turn})
            # Every completed turn, not just the session-wide average - an
            # average alone hides which specific turn was slow and why.
            lines += ["", f"EVERY TURN ({len(self.turn_history)} so far)", "-" * 28]
            for i, turn in enumerate(self.turn_history, start=1):
                parts = [f"{label(k)} {fmt_ms(turn.get(k))}{quality(k, turn.get(k))}" for k in keys]
                lines.append(f"#{i:<3} " + "  ".join(parts))

            lines += ["", "SESSION AVG / MIN / MAX", "-" * 28]
            for k in keys:
                values = [t[k] for t in self.turn_history if k in t]
                lines.append(
                    f"{label(k):<26} avg {fmt_ms(sum(values) / len(values))}"
                    f" | min {fmt_ms(min(values))} | max {fmt_ms(max(values))}"
                )
        return "\n".join(lines)

    async def transcribe(self, sample_rate: int, audio: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = Path(tmp.name)
        try:
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(audio.astype(np.int16).tobytes())
            if self.debug_dir:
                debug_path = self.debug_dir / f"turn_{len(self.turn_history):04d}_{time.time():.0f}.wav"
                debug_path.write_bytes(path.read_bytes())
                duration_s = audio.size / sample_rate
                print(f"[debug] saved captured audio: {debug_path} ({duration_s:.2f}s, rate={sample_rate})")
            resp = await self.client.post(
                f"{self.audiocpp_url}/v1/audio/transcriptions",
                json={"model": "nemotron-asr", "audio": str(path)},
            )
            resp.raise_for_status()
            return resp.json()["text"].strip()
        finally:
            path.unlink(missing_ok=True)

    async def ask_stream(self, text: str) -> AsyncGenerator[str, None]:
        """Streams the LLM reply, yielding one completed sentence at a time
        so the caller can start speaking sentence 1 before sentence 2 even
        exists. Doesn't expose partial, mid-sentence text - that would show
        transcript text before its matching audio exists. Assumes the user
        turn is already appended to self.messages (see add_user_turn), so
        the transcript can show "You: ..." before the LLM call starts.
        """
        buffer = ""
        sentences: list[str] = []
        finish_reason: str | None = None
        # Rebuilt fresh each call (not stored in self.messages) so a
        # language change (see set_language) applies from the next turn.
        current_datetime = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")
        system_content = self.system_prompt_template.format(language=self.language, current_datetime=current_datetime)
        if self.summary:
            # Not stored as a message - it summarizes messages already
            # removed from self.messages (see maybe_compact()).
            system_content += f"\n\nSummary of earlier parts of this conversation: {self.summary}"
        system_message = {"role": "system", "content": system_content}
        async with self.client.stream(
            "POST",
            f"{self.llm_endpoint}/chat/completions",
            # Backstop against a hung generation beyond what n_predict guards.
            timeout=20.0,
            json={
                "model": self.llm_model,
                "messages": [system_message, *self.messages],
                "temperature": 0.7,
                "stream": True,
                # Caps runaway generations - combined with --parallel 1 on
                # the LLM server, one unbounded reply would block every
                # other request behind it. n_predict, not max_tokens: the
                # latter was silently unenforced on this llama-server build.
                "n_predict": 300,
                # Qwen3's thinking-mode toggle (no-op for gpt-oss). Nested
                # under chat_template_kwargs rather than top-level
                # reasoning_effort - the top-level form hung every request
                # on this llama-server build.
                "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"},
            },
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line.removeprefix("data: ")
                if payload == "[DONE]":
                    break
                choice = json.loads(payload)["choices"][0]
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice["delta"].get("content", "")
                if not delta:
                    continue
                buffer += delta.translate(_TTS_UNSAFE_CHARS)
                *complete, buffer = _SENTENCE_END_RE.split(buffer)
                for raw_sentence in complete:
                    sentence = raw_sentence.replace("\n", " ").strip()
                    if sentence:
                        sentences.append(sentence)
                        yield sentence
        tail = buffer.replace("\n", " ").strip()
        if tail and finish_reason == "length":
            # n_predict cut generation off mid-sentence - drop the dangling
            # fragment rather than speak/display it as a real answer.
            print(f"!! reply truncated at n_predict cap, dropping incomplete tail: {tail!r}")
        elif tail:
            sentences.append(tail)
            yield tail
        reply = {"role": "assistant", "content": " ".join(sentences)}
        self.messages.append(reply)
        self.display_messages.append(reply)

    async def speak(self, text: str) -> tuple[int, np.ndarray]:
        """Speaks text via the resident server, honoring inline "[pause Ns]"
        markers (see synthesize_speech_with_pauses above).
        """
        wav_bytes = await synthesize_speech_with_pauses(
            self.client, self.audiocpp_url, text, self.voice_ref, self.exaggeration
        )
        with wave.open(BytesIO(wav_bytes), "rb") as w:
            rate = w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return rate, pcm


def _get_or_create_self_signed_cert() -> tuple[Path, Path]:
    ssl_dir = Path("~/.polyphon-audio/ssl").expanduser()
    ssl_dir.mkdir(parents=True, exist_ok=True)
    cert_file, key_file = ssl_dir / "cert.pem", ssl_dir / "key.pem"
    if not (cert_file.exists() and key_file.exists()):
        cmd = ["openssl", "req"]
        openssl_exe = shutil.which("openssl")
        if openssl_exe:
            openssl_config = Path(openssl_exe).resolve().parent.parent / "ssl" / "openssl.cnf"
            if openssl_config.is_file():
                cmd.extend(["-config", str(openssl_config)])
        cmd.extend([
            "-x509", "-newkey", "rsa:2048",
            "-keyout", str(key_file), "-out", str(cert_file),
            "-days", "365", "-nodes", "-subj", "/CN=PolyphonAudio",
        ])
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"OpenSSL certificate generation failed: {e.stderr.strip()}") from e
    return cert_file, key_file


_LAYOUT_CSS = """
.gradio-container { max-width: 100% !important; }
/* gr.Textbox sizes its box from lines=/max_lines=, which tops out well
   short of the viewport - force a viewport-relative height instead. */
#transcript_box, #stats_box textarea {
    height: calc(100vh - 230px) !important;
    max-height: calc(100vh - 230px) !important;
    overflow-y: auto !important;
    padding: 16px 20px !important;
    box-sizing: border-box !important;
}
/* Row children stretch by default, so the transcript column's forced
   near-full-viewport height was stretching the mic column to match,
   pushing the WebRTC widget's controls below the fold. */
#main_row {
    align-items: flex-start !important;
}
/* full_screen=False on the WebRTC() call below already stops it using
   position:fixed/.full-screen; this just sets a sane minimum height. */
#webrtc_box {
    min-height: 120px;
}
"""

# Auto-scrolls both panels as content streams in, only when already near
# the bottom. Polls the stats textarea (Gradio sets .value as a DOM
# property, firing no mutation event); a MutationObserver drives the
# transcript panel (gr.HTML), which does mutate the DOM on update.
_AUTOSCROLL_HEAD = """
<script>
function scrollIfNearBottom(el) {
  if (!el) return;
  const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  if (nearBottom) el.scrollTop = el.scrollHeight;
}

setInterval(() => {
  const statsWrapper = document.getElementById("stats_box");
  scrollIfNearBottom(statsWrapper && statsWrapper.querySelector("textarea"));
}, 200);

function setupTranscriptObserver() {
  const el = document.getElementById("transcript_box");
  if (!el) {
    setTimeout(setupTranscriptObserver, 200);
    return;
  }
  new MutationObserver(() => scrollIfNearBottom(el)).observe(el, { childList: true, subtree: true });
}
setupTranscriptObserver();

// FastRTC's device-picker icons have aria-label but no visible tooltip;
// copy it into title. Polls since elements render async; no-op once set.
setInterval(() => {
  document.querySelectorAll("[aria-label]:not([title])").forEach((el) => {
    el.title = el.getAttribute("aria-label");
  });
}, 500);
</script>
"""


def build_live_chat_ui(conversation: Conversation) -> tuple[gr.Dropdown, gr.Markdown]:
    """Lays out the Live Chat UI (WebRTC stream, transcript, latency panel,
    language/voice dropdowns) and wires its events into whatever
    gr.Blocks/gr.Tab context is currently active - not tied to its own
    gr.Blocks()/demo.launch(), so scripts/studio.py can reuse this exact
    loop from inside a gr.Tab().

    Returns the voice dropdown and readiness indicator so the caller can
    wire a `demo.load()` refresh for both - choices= is fixed at build
    time, so a voice saved later wouldn't otherwise reach a freshly-loaded
    session.
    """

    async def respond(
        audio: tuple[int, np.ndarray], *extra: object
    ) -> AsyncGenerator[tuple[int, np.ndarray] | AdditionalOutputs, None]:
        # ReplyOnPause calls this with an extra positional arg beyond the
        # webrtc audio; *extra just absorbs it (unread). async, not sync, so
        # FastRTC can actually cancel an in-flight reply on interruption
        # (see Conversation.client's comment).
        del extra

        t0 = time.monotonic()
        sample_rate, array = audio
        # Peak amplitude distinguishes "mic captured nothing" (near 0) from
        # "sound arrived but ASR heard nothing" (transcript still empty
        # below) in the logs, without needing the debug_dir recording.
        peak = float(np.abs(array).max()) if array.size else 0.0
        print(f"-> respond() called: {array.shape} samples @ {sample_rate}Hz, peak amplitude {peak:.0f}")
        transcript = await conversation.transcribe(sample_rate, array)
        t1 = time.monotonic()
        if not transcript:
            print("-> transcribe() returned empty - ASR heard nothing usable in this turn")
            return
        asr_s = t1 - t0
        print(f"-> heard: {transcript!r}  [asr {asr_s:.2f}s]")
        conversation.add_user_turn(transcript)
        yield AdditionalOutputs(conversation.transcript_html(), conversation.stats_report({"asr": asr_s}))

        sentences = []
        t_prev = t1
        ttfb_s = None
        try:
            async for sentence in conversation.ask_stream(transcript):
                t_llm_done = time.monotonic()
                sentences.append(sentence)
                audio_out = await conversation.speak(sentence)
                t_tts_done = time.monotonic()
                if ttfb_s is None:
                    ttfb_s = t_tts_done - t0
                llm_s, tts_s = t_llm_done - t_prev, t_tts_done - t_llm_done
                print(f"   sentence={sentence!r}  [llm +{llm_s:.2f}s, tts {tts_s:.2f}s]")
                t_prev = t_tts_done
                live = {"asr": asr_s, "ttfb": ttfb_s, "last_sentence_llm": llm_s, "last_sentence_tts": tts_s}
                # Audio and its matching transcript text go out back-to-back,
                # right as this sentence's audio is about to play.
                yield audio_out
                yield AdditionalOutputs(
                    conversation.transcript_html(spoken_this_turn=" ".join(sentences)),
                    conversation.stats_report(live),
                )
        except httpx.HTTPError as e:
            # Report a timeout/connection error in the UI instead of letting
            # it silently kill the whole response generator.
            print(f"!! LLM/TTS request failed mid-turn: {e!r}")
            yield AdditionalOutputs(
                conversation.transcript_html(spoken_this_turn=" ".join(sentences)),
                f"ASR: {asr_s * 1000:.0f}ms\nERROR: {e}",
            )
            return

        # Empty/filtered LLM reply, or the turn got interrupted mid-stream -
        # nothing to speak or record as a completed turn.
        if not sentences:
            yield AdditionalOutputs(conversation.transcript_html(), conversation.stats_report({"asr": asr_s}))
            return

        reply = " ".join(sentences)
        total_s = time.monotonic() - t0
        print(f"<- {reply!r}  [total {total_s:.2f}s]")
        # ask_stream() already appended the assistant turn to self.messages.
        assert ttfb_s is not None
        conversation.record_turn(prompt=transcript, reply=reply, asr=asr_s, ttfb=ttfb_s, total=total_s)
        # After the turn, not before/during - preparing for the next turn
        # only, never adding latency to the one the user is waiting on.
        await conversation.maybe_compact()
        final_stats = conversation.stats_report({"asr": asr_s, "ttfb": ttfb_s, "total": total_s})
        yield AdditionalOutputs(conversation.transcript_html(), final_stats)

    with gr.Row(elem_id="main_row"):
        with gr.Column(scale=1, min_width=260):
            # Language dropdown above the mic control - flex-start on
            # #main_row (see _LAYOUT_CSS) keeps this column's natural
            # height instead of stretching to match the transcript column.
            readiness_ui = gr.Markdown("Checking audio.cpp/LLM status...")
            language_ui = gr.Dropdown(
                label="Response language",
                choices=LANGUAGE_CHOICES,
                value=conversation.language,
            )
            _registry = load_voice_registry()
            _default_voice = next((v for v in _registry if v["path"] == conversation.voice_ref), None)
            if _default_voice is None and _registry:
                # Falls back to (and actually applies) the first saved voice
                # if the configured voice_ref isn't in the registry, so the
                # dropdown matches what Live Chat actually speaks with.
                _default_voice = _registry[0]
                conversation.set_voice_ref(
                    _default_voice["path"], _default_voice.get("exaggeration", DEFAULT_EXAGGERATION)
                )
            voice_ui = gr.Dropdown(
                label="Persona voice",
                choices=voice_dropdown_choices(_registry),
                value=_default_voice["id"] if _default_voice else None,
            )
            webrtc = WebRTC(
                label="Stream",
                mode="send-receive",
                modality="audio",
                elem_id="webrtc_box",
                # "ideal", not fastrtc's default "exact": an unsupported
                # mic/driver then makes getUserMedia() fail closed with no
                # visible error - just a permanently unconnected mic and a
                # waveform that never moves. "ideal" lets the browser fall
                # back instead.
                track_constraints={
                    "echoCancellation": True,
                    "noiseSuppression": {"ideal": True},
                    "autoGainControl": {"ideal": True},
                    "sampleRate": {"ideal": 24000},
                    "sampleSize": {"ideal": 16},
                    "channelCount": {"ideal": 1},
                },
                # Stops the component using its position:fixed .full-screen
                # layout, which otherwise covers the page and swallows
                # clicks. variant="textbox" was tried but swaps in a
                # different chat-input widget entirely, not wanted here.
                full_screen=False,
            )
        with gr.Column(scale=3):
            # gr.HTML, not gr.Textbox: transcript_html() bolds each speaker
            # label. Output-only, never in inputs=.
            transcript_ui = gr.HTML(label="Live transcript", elem_id="transcript_box")
        with gr.Column(scale=1, min_width=260):
            stats_ui = gr.Textbox(
                label="Latency", interactive=False, lines=24, max_lines=40, elem_id="stats_box"
            )

    # transcript_ui/stats_ui are output-only (rendered fresh from
    # conversation.messages each call), so inputs=[webrtc] is deliberately
    # the only entry despite respond() taking an extra positional arg.
    webrtc.stream(
        fn=ReplyOnPause(
            respond,
            # ReplyOnPause's cutoff is per-chunk, not a rolling silence
            # timer: below the 0.6s default, speech fragmented at exactly
            # 2x chunk duration. 3s fixed fragmentation but made
            # interruption sluggish, since the same window gates "user
            # paused, interrupt the reply". 1.2s balances both.
            algo_options=AlgoOptions(audio_chunk_duration=1.2),
        ),
        inputs=[webrtc],
        outputs=[webrtc],
        # Must be set here, not as a post-construction attribute - stream()
        # overwrites it from its own default (1) regardless. >1 matters
        # because fastrtc never frees a slot for an abandoned/disconnected
        # tab, so 1 would lock out every other session until restart.
        concurrency_limit=3,
    )
    webrtc.on_additional_outputs(
        lambda transcript, stats: (transcript, stats),
        outputs=[transcript_ui, stats_ui],
        # queue=False let per-sentence update bursts apply out of order,
        # showing the transcript intermittently revert to an older snapshot.
        queue=True,
        show_progress="hidden",
    )
    # Own event, not folded into respond()'s inputs= - just updates the
    # Conversation object directly, no output of its own.
    language_ui.change(conversation.set_language, inputs=[language_ui], outputs=[])

    def apply_voice_choice(voice_id: str | None) -> None:
        # Looks up the path fresh from disk (not the id the dropdown holds)
        # in case the registry changed since UI-build time.
        if not voice_id:
            return
        voice = next((v for v in load_voice_registry() if v["id"] == voice_id), None)
        if voice:
            conversation.set_voice_ref(voice["path"], voice.get("exaggeration", DEFAULT_EXAGGERATION))

    voice_ui.change(apply_voice_choice, inputs=[voice_ui], outputs=[])
    return voice_ui, readiness_ui


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--audiocpp-url", default=DEFAULT_AUDIOCPP_URL)
    parser.add_argument("--llm-endpoint", default=DEFAULT_LLM_ENDPOINT)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--voice-ref", default=DEFAULT_VOICE_REF)
    parser.add_argument(
        "--system-prompt-template",
        default=DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        help=(
            "Must contain {language} (filled from the UI dropdown) and {current_datetime} "
            "(filled with the real current date/time on every call)"
        ),
    )
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, choices=LANGUAGE_CHOICES)
    parser.add_argument("--self-signed", action="store_true", help="Serve HTTPS with an auto-generated cert")
    parser.add_argument(
        "--debug-audio-dir",
        type=Path,
        default=None,
        help="Save each captured utterance's WAV here, to check what was actually recorded vs transcribed",
    )
    args = parser.parse_args()

    conversation = Conversation(
        debug_dir=args.debug_audio_dir,
        audiocpp_url=args.audiocpp_url,
        llm_endpoint=args.llm_endpoint,
        llm_model=args.llm_model,
        voice_ref=args.voice_ref,
        system_prompt_template=args.system_prompt_template,
        language=args.language,
    )

    with gr.Blocks(css=_LAYOUT_CSS, head=_AUTOSCROLL_HEAD, title="Polyphon Audio - Live Chat") as demo:
        gr.HTML("<h1 style='text-align:center'>Polyphon Audio — Live Chat</h1>")
        voice_ui, readiness_ui = build_live_chat_ui(conversation)

        # See build_live_chat_ui()'s own docstring for why this needs to be
        # re-read per session rather than trusting the choices= it was
        # built with - a voice saved (e.g. via scripts/studio.py, pointed
        # at the same voices/ directory) after this process started
        # wouldn't otherwise reach a later or freshly-reloaded browser tab.
        # The readiness check is re-run for the same reason (a fresh session
        # deserves a fresh answer, not whatever was true when the process
        # started), not because it's fixed at build time like choices= is.
        async def refresh():
            voice_update = gr.update(choices=voice_dropdown_choices(load_voice_registry()))
            return voice_update, await conversation.check_readiness()

        demo.load(refresh, outputs=[voice_ui, readiness_ui])

    launch_kwargs = {"server_name": args.host, "server_port": args.port}
    if args.self_signed:
        cert_path, key_path = _get_or_create_self_signed_cert()
        launch_kwargs["ssl_certfile"] = str(cert_path)
        launch_kwargs["ssl_keyfile"] = str(key_path)
        launch_kwargs["ssl_verify"] = False
        print(f"HTTPS enabled with a self-signed cert ({cert_path}). Accept the browser warning on first visit.")

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
