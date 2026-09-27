"""Polyphon Audio Studio: a four-tab web UI on top of the same local
services the rest of this project already uses.

- **Live Chat** - the real-time voice loop from `scripts/realtime/bot_fastrtc.py`,
  reused as-is (imports `build_live_chat_ui`/`Conversation`, not a fork of
  them) so there's exactly one implementation of that loop, not two to keep
  in sync.
- **Text to Speech** - batch synthesis from typed text, picking a voice from
  the library, with a downloadable clip. The ElevenLabs "Speech Synthesis"
  equivalent - the thing that makes the project's own "private alternative
  to cloud TTS" pitch (see README) actually usable without editing scripts.
- **Voice Library** - design a brand-new synthetic voice from a text
  description (`audio.cpp`'s `qwen3_tts` "vdes" task, wired into
  `audiocpp_server` via a `server.json` model entry - see README
  "Quickstart"), preview it, and save it under a name; saved voices
  immediately show up in the Text to Speech tab's picker.
- **Sound & Music** - short text-to-music or text-to-sound-effect clips
  (`audio.cpp`'s `stable_audio` family, task "gen" - see README
  "Quickstart"), for game audio, video stingers, or ambience.

All four tabs hit the same resident `audiocpp_server` (and, for Live Chat,
the same local LLM) - nothing here starts a second copy of any model.
Voices are a small on-disk registry (`voices/registry.json`), not a
database - this is a single-user local tool, not a multi-tenant service.

Prerequisites (same as `bot_fastrtc.py` - see README "Quickstart"):
  1. audiocpp_server running, with "voice-design" (qwen3_tts, task vdes)
     and "sound-music"/"sound-sfx" (stable_audio, task gen) model entries
     in server.json alongside chatterbox/nemotron-asr.
  2. A local OpenAI-compatible LLM server, for the Live Chat tab only.

Run: .venv-fastrtc/bin/python scripts/studio.py --self-signed
Then open the printed https://<host>:<port> URL in a browser.
"""

from __future__ import annotations

import argparse
import base64
import sys
import uuid
from pathlib import Path

import gradio as gr
import httpx
from dotenv import load_dotenv

# bot_fastrtc.py lives in scripts/realtime/, not on sys.path by default.
sys.path.insert(0, str(Path(__file__).resolve().parent / "realtime"))
import bot_fastrtc  # noqa: E402

load_dotenv()

_OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "studio_tts"
_VOICES_DIR = bot_fastrtc.VOICES_DIR

# Registry helpers live in bot_fastrtc.py (shared with Live Chat's voice
# picker), not duplicated here.
_load_registry = bot_fastrtc.load_voice_registry
_save_registry = bot_fastrtc.save_voice_registry
_voice_dropdown_choices = bot_fastrtc.voice_dropdown_choices


def build_tts_tab(client: httpx.AsyncClient, audiocpp_url: str) -> gr.Dropdown:
    """Builds the Text to Speech tab. Returns its voice-picker component so
    the Voice Library tab can add it as an extra output and refresh it the
    moment a new voice is saved, without the two tabs needing any shared
    state beyond the on-disk registry both already read from.
    """
    with gr.Tab("Text to Speech"):
        gr.Markdown(
            "Generate speech from text using a saved voice from the Voice Library tab, "
            "then download the clip - for narration, video voiceover, or anything else "
            "outside a live conversation."
        )
        text_in = gr.Textbox(label="Text", lines=8, placeholder="Type or paste the text to speak...")
        voice_dd = gr.Dropdown(label="Voice", choices=_voice_dropdown_choices(_load_registry()))
        generate_btn = gr.Button("Generate", variant="primary")
        # show_download_button=True, not the default: a non-interactive
        # gr.Audio otherwise shows no download control at all.
        audio_out = gr.Audio(label="Result", interactive=False, show_download_button=True)

        async def generate(text: str, voice_id: str | None):
            if not text.strip():
                raise gr.Error("Enter some text first.")
            if not voice_id:
                raise gr.Error("Pick a voice first - design or save one in the Voice Library tab.")
            voice = next((v for v in _load_registry() if v["id"] == voice_id), None)
            if voice is None:
                raise gr.Error("That voice is no longer in the library - pick another.")
            # synthesize_speech_with_pauses (not a raw /v1/audio/speech call)
            # so a "[pause 0.6s]" marker splices real silence instead of
            # getting read aloud literally - same handling Live Chat gets.
            wav_bytes = await bot_fastrtc.synthesize_speech_with_pauses(
                client,
                audiocpp_url,
                text,
                voice["path"],
                voice.get("exaggeration", bot_fastrtc.DEFAULT_EXAGGERATION),
            )
            _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = _OUTPUTS_DIR / f"{uuid.uuid4().hex}.wav"
            out_path.write_bytes(wav_bytes)
            return str(out_path)

        generate_btn.click(generate, inputs=[text_in, voice_dd], outputs=[audio_out])
    return voice_dd


def build_voice_library_tab(
    client: httpx.AsyncClient, audiocpp_url: str, voice_pickers: list[gr.Dropdown]
) -> tuple[gr.Dataframe, gr.Dropdown]:
    """Design a new voice from a text description (no reference recording -
    see README "Quickstart" for the underlying qwen3_tts/vdes wiring), preview
    it, and save it into the registry. `voice_pickers` is every other tab's
    own voice dropdown (Text to Speech, Live Chat) - passed in so
    saving/renaming/deleting a voice here refreshes all of them immediately.
    Returns this tab's own table and manage-voice dropdown so main() can
    include them in the same demo.load() refresh as voice_pickers, since
    their `value=` is otherwise fixed at Blocks-build time.
    """
    with gr.Tab("Voice Library"):
        gr.Markdown(
            "Describe a voice in plain language and preview it - no recording needed, "
            "and no real person's voice anywhere in the chain (see README \"Voice design\"). "
            "Save one you like to use it in the Text to Speech tab."
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Design a new voice")
                instruction_in = gr.Textbox(
                    label="Voice description",
                    placeholder="A calm, warm, middle-aged male narrator, clear articulation, slight British accent",
                )
                preview_text_in = gr.Textbox(
                    label="Preview text",
                    lines=3,
                    # Longer and more prosodically varied than a flat sentence -
                    # this becomes the reference clip Chatterbox clones from
                    # later (see save_voice()), and a flat clip reads as robotic
                    # regardless of the "exaggeration" slider.
                    value=(
                        "Hello there! It's wonderful to finally speak with you. "
                        "Can you believe how quickly the day has gone by? "
                        "I've got so much to tell you about, if you have a moment."
                    ),
                )
                exaggeration_in = gr.Slider(
                    label="Expressiveness",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=bot_fastrtc.DEFAULT_EXAGGERATION,
                    info="Higher = more animated delivery when this voice speaks later; lower = flatter/calmer.",
                )
                preview_btn = gr.Button("Generate preview")
                preview_audio = gr.Audio(label="Preview", interactive=False)
                # The just-generated preview clip's path - kept server-side
                # in a gr.State rather than re-derived from preview_audio's
                # value, since that value is a URL Gradio serves the file
                # under, not the on-disk path save_voice() needs to copy.
                preview_path_state = gr.State(None)
                name_in = gr.Textbox(label="Save as", placeholder="e.g. Warm Narrator")
                save_btn = gr.Button("Save to library", variant="primary")
                save_status = gr.Markdown()
            with gr.Column():
                gr.Markdown("### Your voices")
                library_table = gr.Dataframe(
                    headers=["Label", "Source"],
                    value=[[v["label"], v["source"]] for v in _load_registry()],
                    interactive=False,
                )
                gr.Markdown("### Rename or delete a voice")
                manage_voice_dd = gr.Dropdown(
                    label="Select a voice to manage", choices=_voice_dropdown_choices(_load_registry())
                )
                rename_in = gr.Textbox(label="New name", placeholder="e.g. Warm Narrator")
                with gr.Row():
                    rename_btn = gr.Button("Rename")
                    delete_btn = gr.Button("Delete", variant="stop")
                manage_status = gr.Markdown()

        # Shared by save/rename/delete below - each refreshes the same set
        # of components (every voice dropdown, plus this tab's own table).
        def _refresh_pickers(voices: list[dict], manage_value: str | None = None):
            picker_update = gr.update(choices=_voice_dropdown_choices(voices))
            manage_update = gr.update(choices=_voice_dropdown_choices(voices), value=manage_value)
            table_rows = [[v["label"], v["source"]] for v in voices]
            return [picker_update] * len(voice_pickers), manage_update, table_rows

        async def generate_preview(instruction: str, preview_text: str):
            if not instruction.strip():
                raise gr.Error("Describe the voice you want first.")
            resp = await client.post(
                f"{audiocpp_url}/v1/audio/speech",
                json={"model": "voice-design", "input": preview_text, "instructions": instruction},
            )
            resp.raise_for_status()
            _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            preview_path = _OUTPUTS_DIR / f"preview_{uuid.uuid4().hex}.wav"
            preview_path.write_bytes(resp.content)
            return str(preview_path), str(preview_path)

        preview_btn.click(
            generate_preview, inputs=[instruction_in, preview_text_in], outputs=[preview_audio, preview_path_state]
        )

        def save_voice(preview_path: str | None, name: str, exaggeration: float):
            if not preview_path:
                raise gr.Error("Generate a preview first.")
            if not name.strip():
                raise gr.Error("Give the voice a name to save it under.")
            voices = _load_registry()
            voice_id = uuid.uuid4().hex[:8]
            saved_path = _VOICES_DIR / f"{voice_id}.wav"
            _VOICES_DIR.mkdir(parents=True, exist_ok=True)
            saved_path.write_bytes(Path(preview_path).read_bytes())
            voices.append(
                {
                    "id": voice_id,
                    "label": name.strip(),
                    "path": str(saved_path),
                    "source": "designed",
                    "exaggeration": exaggeration,
                }
            )
            _save_registry(voices)
            picker_updates, manage_update, table_rows = _refresh_pickers(voices)
            return (
                f"Saved **{name.strip()}** to the library.",
                *picker_updates,
                manage_update,
                table_rows,
            )

        save_btn.click(
            save_voice,
            inputs=[preview_path_state, name_in, exaggeration_in],
            outputs=[save_status, *voice_pickers, manage_voice_dd, library_table],
        )

        def rename_voice(voice_id: str | None, new_name: str):
            if not voice_id:
                raise gr.Error("Pick a voice to rename first.")
            if not new_name.strip():
                raise gr.Error("Enter a new name.")
            voices = _load_registry()
            voice = next((v for v in voices if v["id"] == voice_id), None)
            if voice is None:
                raise gr.Error("That voice is no longer in the library.")
            voice["label"] = new_name.strip()
            _save_registry(voices)
            picker_updates, manage_update, table_rows = _refresh_pickers(voices, manage_value=voice_id)
            return f"Renamed to **{new_name.strip()}**.", *picker_updates, manage_update, table_rows

        rename_btn.click(
            rename_voice,
            inputs=[manage_voice_dd, rename_in],
            outputs=[manage_status, *voice_pickers, manage_voice_dd, library_table],
        )

        def delete_voice(voice_id: str | None):
            if not voice_id:
                raise gr.Error("Pick a voice to delete first.")
            voices = _load_registry()
            voice = next((v for v in voices if v["id"] == voice_id), None)
            if voice is None:
                raise gr.Error("That voice is no longer in the library.")
            voices = [v for v in voices if v["id"] != voice_id]
            _save_registry(voices)
            Path(voice["path"]).unlink(missing_ok=True)
            picker_updates, manage_update, table_rows = _refresh_pickers(voices)
            return f"Deleted **{voice['label']}**.", *picker_updates, manage_update, table_rows

        delete_btn.click(
            delete_voice,
            inputs=[manage_voice_dd],
            outputs=[manage_status, *voice_pickers, manage_voice_dd, library_table],
        )

    return library_table, manage_voice_dd


def build_sound_tab(client: httpx.AsyncClient, audiocpp_url: str) -> None:
    """Text-to-music and text-to-sound-effect generation (`audio.cpp`'s
    `stable_audio` family, task "gen" - see README "Quickstart"), wired to
    a "sound-music"/"sound-sfx" pair of server.json model entries. Unlike
    the other tabs' `/v1/audio/speech`, this goes through the generic
    `/v1/tasks/run` endpoint and gets base64-encoded audio back in a JSON
    body - stable_audio isn't a speech model.
    """
    with gr.Tab("Sound & Music"):
        gr.Markdown(
            "Generate short music clips or sound effects from a text description - "
            "useful for game audio, video stingers, or ambience, without hunting "
            "stock-audio libraries for a license that fits."
        )
        kind_dd = gr.Dropdown(
            label="Kind",
            choices=[("Music", "sound-music"), ("Sound effect", "sound-sfx")],
            value="sound-music",
        )
        text_in = gr.Textbox(
            label="Description",
            placeholder="uplifting house music with bright synths and festival drums",
        )
        negative_in = gr.Textbox(
            label="Negative prompt (optional)",
            value="",
            placeholder="vocals, distortion, low quality",
            info="Describe what to avoid - stable_audio supports this natively.",
        )
        duration_in = gr.Slider(
            label="Duration (seconds)",
            minimum=1,
            maximum=60,
            step=1,
            value=15,
            info="Longer clips take proportionally longer to generate.",
        )
        generate_btn = gr.Button("Generate", variant="primary")
        audio_out = gr.Audio(label="Result", interactive=False, show_download_button=True)

        async def generate(kind: str, text: str, negative: str, duration: float):
            if not text.strip():
                raise gr.Error("Describe the music or sound effect first.")
            payload = {"model": kind, "text": text, "duration_seconds": duration}
            if negative.strip():
                payload["options"] = {"negative_prompt": negative.strip()}
            resp = await client.post(f"{audiocpp_url}/v1/tasks/run", json=payload)
            resp.raise_for_status()
            wav_bytes = base64.b64decode(resp.json()["audio"])
            _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = _OUTPUTS_DIR / f"{uuid.uuid4().hex}.wav"
            out_path.write_bytes(wav_bytes)
            return str(out_path)

        generate_btn.click(
            generate, inputs=[kind_dd, text_in, negative_in, duration_in], outputs=[audio_out]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--audiocpp-url", default=bot_fastrtc.DEFAULT_AUDIOCPP_URL)
    parser.add_argument("--llm-endpoint", default=bot_fastrtc.DEFAULT_LLM_ENDPOINT)
    parser.add_argument("--llm-model", default=bot_fastrtc.DEFAULT_LLM_MODEL)
    parser.add_argument("--voice-ref", default=bot_fastrtc.DEFAULT_VOICE_REF)
    parser.add_argument("--system-prompt-template", default=bot_fastrtc.DEFAULT_SYSTEM_PROMPT_TEMPLATE)
    parser.add_argument("--language", default=bot_fastrtc.DEFAULT_LANGUAGE, choices=bot_fastrtc.LANGUAGE_CHOICES)
    parser.add_argument("--self-signed", action="store_true", help="Serve HTTPS with an auto-generated cert")
    parser.add_argument("--debug-audio-dir", type=Path, default=None)
    args = parser.parse_args()

    conversation = bot_fastrtc.Conversation(
        debug_dir=args.debug_audio_dir,
        audiocpp_url=args.audiocpp_url,
        llm_endpoint=args.llm_endpoint,
        llm_model=args.llm_model,
        voice_ref=args.voice_ref,
        system_prompt_template=args.system_prompt_template,
        language=args.language,
    )
    # Separate from Conversation's own client - the other tabs' TTS/voice-design
    # calls have nothing to do with a Live Chat session.
    client = httpx.AsyncClient(timeout=60.0)

    with gr.Blocks(
        css=bot_fastrtc._LAYOUT_CSS, head=bot_fastrtc._AUTOSCROLL_HEAD, title="Polyphon Audio Studio"
    ) as demo:
        gr.HTML("<h1 style='text-align:center'>Polyphon Audio Studio</h1>")
        with gr.Tabs():
            with gr.Tab("Live Chat"):
                # No visibility guard needed: the WebRTC connection only
                # opens once the user presses Record, which requires this
                # tab to already be selected.
                live_chat_voice_dd, readiness_ui = bot_fastrtc.build_live_chat_ui(conversation)
            tts_voice_dd = build_tts_tab(client, args.audiocpp_url)
            library_table, manage_voice_dd = build_voice_library_tab(
                client, args.audiocpp_url, [tts_voice_dd, live_chat_voice_dd]
            )
            build_sound_tab(client, args.audiocpp_url)

        # Every dropdown's choices= was fixed at Blocks-build time, so a
        # freshly-loaded/different browser tab wouldn't otherwise see a
        # voice saved after this process started. demo.load() re-reads the
        # registry per session to keep every fresh tab in sync with disk.
        voice_dropdowns = [tts_voice_dd, live_chat_voice_dd, manage_voice_dd]

        async def refresh_all():
            voices = _load_registry()
            dropdown_updates = [gr.update(choices=_voice_dropdown_choices(voices))] * len(voice_dropdowns)
            table_rows = [[v["label"], v["source"]] for v in voices]
            return [*dropdown_updates, table_rows, await conversation.check_readiness()]

        demo.load(refresh_all, outputs=[*voice_dropdowns, library_table, readiness_ui])

    launch_kwargs = {"server_name": args.host, "server_port": args.port}
    if args.self_signed:
        cert_path, key_path = bot_fastrtc._get_or_create_self_signed_cert()
        launch_kwargs["ssl_certfile"] = str(cert_path)
        launch_kwargs["ssl_keyfile"] = str(key_path)
        launch_kwargs["ssl_verify"] = False
        print(f"HTTPS enabled with a self-signed cert ({cert_path}). Accept the browser warning on first visit.")

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
