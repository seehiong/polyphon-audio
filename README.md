# Polyphon Audio

**Local text-to-speech and voice cloning**
*A private, offline alternative to cloud TTS services — for audiobooks, video narration, e-learning, and voice-first apps.*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Offline First](https://img.shields.io/badge/Privacy-100%25_Offline-success.svg)]()

---

## What this does

A local equivalent of ElevenLabs-style TTS, running entirely on your own
hardware — no cloud calls, no per-character billing, no internet
dependency once the models are downloaded:

- **Voice cloning** — clone a voice from a short reference clip, then
  generate any text in that voice.
- **Multi-voice narration & dialogue** — script lines for multiple named
  voices and get one stitched audio file back. Inline `[pause 0.6s]`
  markers in a line's text splice in real silence at that point.
- **Voice design** — generate a brand-new voice from a text description
  alone, no recording needed at all.
- **Sound effects & music** — generate short clips from a text
  description, for game audio, video stingers, or ambience.
- **A conversational voice-persona loop** — wire in a local LLM so the
  system can listen, think, and speak back in a chosen voice.
- **A web UI** (`scripts/studio.py`) — Live Chat, Text to Speech, Voice
  Library, and Sound & Music tabs, each with a downloadable clip, plus a
  Models tab to see and unload what's in VRAM - or use the scripts directly.

Built on [`audio.cpp`](https://github.com/0xShug0/audio.cpp) (a native
`ggml`-based inference engine, no Python runtime needed to run models),
which supports NVIDIA, AMD, Apple Silicon, and CPU-only machines.

## Prerequisites

- **Linux with an AMD GPU (ROCm/HIP):** `scripts/setup_audiocpp.sh` uses
  this project's AMD/ROCm/Ubuntu-specific build flags.
- **Windows:** `scripts/setup_audiocpp.bat [cpu|cuda|vulkan]` uses
  audio.cpp's Windows build helper. Install Visual Studio Build Tools 2022+
  with the C++ desktop workload, Windows SDK, CMake, Ninja, and MSVC OpenMP
  components; CUDA or Vulkan SDK is also required for that selected backend.
- On NVIDIA/Apple Silicon/CPU systems not covered above, build `audio.cpp`
  yourself following its own README, then point the scripts below at the
  resulting binaries (`--cli`/`--backend`) - everything past step 1 is
  backend-agnostic.
- `git`, `cmake`, and Python 3.10+. The Linux AMD path also needs a ROCm/HIP
  install with `hipconfig` on `PATH`.
- `openssl` on `PATH` when using `--self-signed` for local HTTPS.
- Python 3.10+ and [`uv`](https://docs.astral.sh/uv/) (creates this
  project's venvs; also runs one-off tools like `ruff` with no persistent
  install).
- ~10GB free disk for `audio.cpp`'s build plus its downloaded models
  (Quickstart step 1).
- Only if you want the conversational Live Chat feature (Quickstart step
  3): your own local OpenAI-compatible LLM server - `llama-server` from
  [`llama.cpp`](https://github.com/ggml-org/llama.cpp) is what this README
  uses - and the `hf` CLI (`pip install -U "huggingface_hub[cli]"`) to
  download model weights.

## Quickstart

Everything below is the *full* live/conversational stack - three
processes running together, plus a one-time build. If all you want is
one-off narration or voice cloning with no live conversation, skip to
"just the batch scripts" at the end of this section - it needs step 1
only, not steps 2-5.

**1. Build `audio.cpp` once.**

```bash
scripts/setup_audiocpp.sh
```

On Windows, run the batch setup from Command Prompt (the default backend is
CPU; CUDA and Vulkan builds can be selected explicitly):

```bat
scripts\setup_audiocpp.bat cuda
```

Clones `audio.cpp` to `~/github/audio.cpp` (kept alongside this project
rather than inside it, same as `llama.cpp`), symlinks it in as
`vendor/audio.cpp`, builds it, and installs the five models this
project's `server.json` expects (~9GB total). The flags it uses are
this project's own AMD/ROCm/Ubuntu-specific workarounds - on
NVIDIA/Apple Silicon/CPU, follow `audio.cpp`'s own README instead
(`scripts/setup_audiocpp.sh` is the reference for which models this
project needs either way). Safe to re-run - skips the clone if
`~/github/audio.cpp` already exists.

The Windows script clones to `%USERPROFILE%\github\audio.cpp`, creates
`vendor\audio.cpp` as a directory junction, builds the selected Windows
backend, and installs the same five models. It copies this project's
`server.json` into the vendor directory and adjusts that copy's backend to
match the build; the tracked root config is left unchanged.

**2. Start `audiocpp_server`** (resident - keeps models loaded instead of
reloading per call), configured by this project's own `server.json`
(tracked at the repo root - `scripts/setup_audiocpp.sh` symlinks it into
`vendor/audio.cpp/server.json` so its `"path"` entries resolve against
`audio.cpp`'s `models/` directory):

`"backend": "hip"` is the Linux AMD path from Prerequisites above. The
Windows setup script writes `"cpu"`, `"cuda"`, or `"vulkan"` into its vendor
copy to match the selected build.

```bash
cd vendor/audio.cpp && ./build_hip/bin/audiocpp_server --config server.json
```

On Windows, from the repository root:

```bat
vendor\audio.cpp\build\windows-cuda-release\bin\audiocpp_server.exe --config vendor\audio.cpp\server.json
```

The model-spec messages may warn that some downloaded GGUFs contain legacy
metadata. They are non-fatal when the server proceeds to `listening` as
shown above; audio.cpp uses its current runtime contract for validation.

**Which models each Studio tab uses.** Only Live Chat's two models are
loaded at server start (`"lazy": false` in `server.json`), so the first
conversation turn doesn't pay a multi-second model load. The rest load the
first time their tab is used, then stay loaded until you unload them from
the Studio's **Models** tab (an unloaded model reloads by itself on its next
use, so unloading never breaks a tab). VRAM figures are measured on CUDA
(q8_0 weights plus runtime buffers) with `scripts/bench_latency.py`. On the
HIP backend (Linux/AMD), the same script's warm latency ran ~25-40% higher
in testing (e.g. ~2.7-3.2s per sentence vs ~2.2s here) - expected, since
`audio.cpp` itself is CUDA-optimized first. Its VRAM columns are not
meaningful on a unified-memory AMD APU (Strix Halo/gfx1151 and similar):
`rocm-smi` tracks a small fixed carve-out there, not the much larger
system-RAM pool HIP actually allocates models from, so the deltas stay near
zero regardless of what loaded.

| `server.json` id | Used by | Loaded | VRAM | Warm latency |
|---|---|---|---|---|
| `nemotron-asr` | Live Chat (speech recognition) | always (at startup) | ~1.1 GB | <0.1 s for an 8 s clip |
| `chatterbox` | Live Chat, Text to Speech | always (at startup) | ~2.4 GB | ~2.2 s per sentence |
| `voice-design` | Voice Library (voice previews) | on demand | ~1.6 GB | ~4.6 s per preview |
| `sound-music` | Sound & Music (music) | on demand | ~1.2 GB | ~0.4 s per 5 s clip |
| `sound-sfx` | Sound & Music (sound effects) | on demand | ~1.2 GB | ~0.4 s per 5 s clip |

All five together need about 7.5 GB, before the LLM. On a smaller GPU,
unload the on-demand models before a Live Chat session ("Unload all except
Live Chat"), and run the LLM on CPU if it doesn't fit alongside ASR and TTS.
On Windows the NVIDIA driver does not fail when VRAM runs out: it silently
spills into system RAM, and measured TTS latency went from ~2.5 s to
30 s+ per sentence. audio.cpp's `max_loaded_models`, `idle_unload_ms`, and
`min_free_memory_mb` settings can also enforce limits automatically - see
its `app/server/README.md`.

**3. Download and start a local LLM** (only needed for the conversational
Live Chat feature, not for narration/voice cloning/TTS alone):

```bash
hf download Qwen/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q4_K_M.gguf --local-dir <your-models-dir>
llama-server --model <your-models-dir>/Qwen3-30B-A3B-Q4_K_M.gguf \
  --alias qwen3-30b-a3b --host 127.0.0.1 --port 8082 --parallel 1
```

**4. Set up the Python environment for the UI scripts** - `.venv-fastrtc`
is what `scripts/realtime/bot_fastrtc.py` and `scripts/studio.py` both
run in; nothing else in this repo needs it:

Linux/macOS:

```bash
uv venv .venv-fastrtc --python 3.12
uv pip install --python .venv-fastrtc/bin/python "fastrtc[vad]" gradio httpx python-dotenv
```

Windows PowerShell:

```powershell
uv venv .venv-fastrtc --python 3.12
uv pip install --python .venv-fastrtc\Scripts\python.exe "fastrtc[vad]" gradio httpx python-dotenv
```

**5. Run the UI.** `scripts/studio.py` (Live Chat + Text to Speech + Voice
Library, one app) is the one to reach for; `bot_fastrtc.py` is the same
Live Chat tab alone, if that's all you want:

Linux/macOS:

```bash
.venv-fastrtc/bin/python scripts/studio.py --self-signed --debug-audio-dir outputs/mic_debug
```

Windows PowerShell:

```powershell
.\.venv-fastrtc\Scripts\python.exe scripts\studio.py --self-signed --debug-audio-dir outputs\mic_debug
```

Open the printed `https://<host>:7862` URL (`--self-signed` is needed
because most browsers require HTTPS for microphone access). `.env` (copy
`.env.example`) overrides `bot_fastrtc.py`/`studio.py`'s default
host/port/model values above if yours differ - the batch scripts below
don't read `.env`; pass `--llm-endpoint`/`--llm-model` etc. directly.

**Just the batch scripts, no live stack at all:** `scripts/narrate.py`
(multi-voice narration/dialogue), `scripts/persona_tts.py` (voice
cloning), and `scripts/persona_turn.py` (one conversational turn, via
`urllib` - stdlib, not `httpx`) all call `audiocpp_cli` directly,
per-invocation, not the resident server. No `audiocpp_server`, no
`.venv-fastrtc` venv, no third-party Python packages at all - just step 1
above and a stock Python 3 interpreter (see each script's own `--help`).
They find the `audiocpp_cli` build the setup scripts produced on their
own - `vendor/audio.cpp/build_hip/bin/audiocpp_cli` (`--backend hip`) or
`vendor/audio.cpp/build/windows-{cuda,vulkan,cpu}-release/bin/audiocpp_cli.exe`
(matching `--backend`), first one found wins; pass `--cli`/`--backend` to
override.
`persona_turn.py`/`persona_session.py`/`persona_meeting.py` still need an
LLM endpoint reachable - their defaults already match step 3 above, so
that alone is enough, without `audiocpp_server` running alongside it.

## Optional integration: other Polyphon projects

Composes with a sibling project, [`polyphon-ai`](https://github.com/seehiong/polyphon-ai)
(local transcription/diarization), when audio output specifically is wanted:

```
audio/video in ──► polyphon-ai ──► named, diarized transcript
                                          │
                                          ▼
                                   (an LLM) ──► response text
                                          │
                                          ▼
                                   polyphon-audio ──► spoken reply,
                                                        per speaker/persona
```

`scripts/persona_meeting.py` takes a `polyphon-ai` transcript JSON and
speaks a reply addressing people by name.

## Repository layout

| Path | What it does |
|---|---|
| `scripts/persona_turn.py` | One conversational turn: text or spoken-audio prompt (`--prompt`/`--prompt-audio`, transcribed via Nemotron ASR) → local LLM → Chatterbox spoken reply. Stateless — each call is a fresh conversation. Also home to the shared helpers (`require`, `synthesize`, `query_llm`, `transcribe_audio`) the other scripts import. |
| `scripts/persona_session.py` | Multi-turn conversation: same loop as `persona_turn.py`, but keeps one growing LLM message history across a `--script` of turns, so later turns can reference earlier ones. Takes `--voices`/`--voice` (named registry) or a plain `--voice-ref`. |
| `scripts/persona_meeting.py` | Feeds a real `polyphon-ai` diarized transcript JSON (`{speakers, segments}`) into the local LLM and speaks its reply — the optional-integration diagram above made concrete. |
| `scripts/narrate.py` | Multi-voice narration/dialogue, no LLM: authored `{"speaker": ..., "text": ...}` lines spoken by named voices from a `--voices` registry, stitched into one clip. |
| `scripts/persona_tts.py` | Thin Chatterbox wrapper matching a conventional XTTS-synthesis CLI interface. |
| `scripts/bench_latency.py` | Benchmarks the resident `audiocpp_server` stack: per-model cold vs warm latency and cumulative VRAM (`nvidia-smi`/`rocm-smi`), a full Live Chat turn (ASR → streamed LLM → first-sentence TTS, i.e. time to first audio), and ASR latency during an abandoned in-flight TTS request (barge-in). Stdlib only; results saved to `outputs/bench/`. On GPUs too small for the LLM plus audio models, run the LLM on CPU - on Windows the driver otherwise silently spills to system RAM and latency collapses. |
| `scripts/record_mic.sh` | `arecord` wrapper (Linux/ALSA only) for capturing a real mic clip in the format `--prompt-audio` expects. On Windows, record a 16kHz mono WAV with any recorder instead. |
| `scripts/setup_audiocpp.sh`, `scripts/setup_audiocpp.bat` | Clone, build, and install the models for `audio.cpp` (Linux AMD/ROCm or Windows CPU/CUDA/Vulkan — see "Quickstart" above). Re-runnable. |
| `scripts/realtime/bot_fastrtc.py` | Live, real-time streaming voice conversation (FastRTC) — mic → VAD → `audiocpp_server` ASR → LLM → `audiocpp_server` TTS → speaker, with barge-in. `--self-signed` for local HTTPS. Requires the `.venv-fastrtc` venv (see "Quickstart" above). |
| `scripts/studio.py` | A 5-tab web UI: Models (which `audiocpp_server` models are loaded, with unload buttons - see the table under "Quickstart" step 2), Live Chat (imports and reuses `bot_fastrtc.py`'s loop, not a fork of it), Text to Speech (batch synthesis, downloadable clip), Voice Library (design a synthetic voice from a text description, save it, use it in the TTS tab), and Sound & Music (text-to-music/SFX clips via `audio.cpp`'s `stable_audio` family). Voices persist in `voices/registry.json` (gitignored). Also requires `.venv-fastrtc`, plus `audiocpp_server` configured with `voice-design` and `sound-music`/`sound-sfx` model entries (see "Quickstart" above). |
| `examples/` | Committed example assets — see `examples/README.md` for what each one is and why. |
| `tests/` | `pytest` coverage for the pure-logic pieces that don't need the GPU, `audio.cpp`, or a running LLM server. Linux/macOS: `uv venv .venv-tests --python 3.12 && uv pip install --python .venv-tests/bin/python pytest && .venv-tests/bin/pytest tests/`. Windows PowerShell: `uv venv .venv-tests --python 3.12; uv pip install --python .venv-tests\Scripts\python.exe pytest; .venv-tests\Scripts\pytest.exe tests\`. |
| `ruff.toml` | Lint config (`uvx ruff check scripts/ tests/` — no project-wide install needed). |
| `vendor/audio.cpp/` | Not committed (gitignored) — a symlink to `~/github/audio.cpp`, the upstream inference engine, built locally per "Quickstart" above. |
| `server.json` | This project's `audiocpp_server` config (models, host/port, backend) — committed here, symlinked into `vendor/audio.cpp/server.json` by `scripts/setup_audiocpp.sh` (copied, with `backend` adjusted, by `scripts/setup_audiocpp.bat` - re-copy after editing it on Windows). |
| `.env.example` | Template for `.env` (gitignored, machine-specific) — which host/port `bot_fastrtc.py`'s `audiocpp_server`/LLM servers are actually running on. Copy it to `.env` and adjust rather than editing the script. |

## License

Apache-2.0 (matching the other Polyphon projects). `audio.cpp` itself is
also Apache-2.0; individual model weights carry their own upstream
licenses (e.g. Chatterbox is MIT) — check each model's card before
depending on it beyond local experimentation.
