"""Local Gradio web UI for Chatterbox Multilingual TTS (English / Polish).

Runs the model on your GPU and serves a browser interface. No HF-Spaces-only
`spaces`/@spaces.GPU bits — it reuses this project's TTSEngine.

Run:
    .venv\\Scripts\\activate
    python app_ui.py
Then open the printed URL (default http://127.0.0.1:7860).
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import logging  # noqa: E402

import gradio as gr  # noqa: E402

from chatterbox_tts import TTSEngine, settings, SUPPORTED  # noqa: E402


class _IgnoreConnReset(logging.Filter):
    """Drop the benign Windows 'ConnectionResetError [WinError 10054]' asyncio
    noise gradio prints when a browser tab/socket closes — not a real error."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(s in msg for s in (
            "ConnectionResetError", "_call_connection_lost", "WinError 10054",
        ))


logging.getLogger("asyncio").addFilter(_IgnoreConnReset())

# Default sample text per language.
DEFAULTS = {
    "en": "Last month we reached two billion views on our YouTube channel.",
    "pl": "W zeszlym miesiacu osiagnelismy dwa miliardy wyswietlen na naszym kanale YouTube.",
    "ru": "В прошлом месяце мы достигли нового рубежа: два миллиарда просмотров на нашем YouTube-канале.",
}

# Load the model once at startup (onto CUDA if available).
ENGINE = TTSEngine()


def _on_lang_change(lang: str) -> str:
    return DEFAULTS.get(lang, "")


def generate(text, language, ref_audio, exaggeration, cfg_weight, temperature, pause_ms):
    if not text or not text.strip():
        raise gr.Error("Please enter some text.")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = settings.output_dir / f"{language}_{ts}.wav"
    path = ENGINE.synthesize_long(
        text.strip(),
        language=language,
        out=out,
        pause_ms=int(pause_ms),
        audio_prompt_path=ref_audio or None,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
    )
    return str(path)


with gr.Blocks(title="Chatterbox TTS — EN/PL") as demo:
    gr.Markdown(
        "# 🗣️ Chatterbox Multilingual TTS — English / Polish\n"
        f"Running on **{ENGINE.device.upper()}**. "
        "Enter text, pick a language, optionally upload a reference voice, then Generate."
    )
    with gr.Row():
        with gr.Column():
            language = gr.Dropdown(
                choices=[(name, code) for code, name in SUPPORTED.items()],
                value="en",
                label="Language",
            )
            text = gr.Textbox(
                value=DEFAULTS["en"],
                label=f"Text (max ~{settings.max_chars} chars)",
                lines=4,
            )
            ref_audio = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference voice (optional — wav/flac/mp3/m4a; match the language)",
            )
            with gr.Accordion("Advanced", open=False):
                exaggeration = gr.Slider(0.25, 2.0, value=settings.exaggeration, step=0.05,
                                         label="Exaggeration (0.5 neutral)")
                cfg_weight = gr.Slider(0.2, 1.0, value=settings.cfg_weight, step=0.05,
                                       label="CFG / pace (0 = language transfer)")
                temperature = gr.Slider(0.05, 2.0, value=settings.temperature, step=0.05,
                                        label="Temperature")
                pause_ms = gr.Slider(0, 1500, value=400, step=50,
                                     label="Pause between sentences (ms) — [pause:N] or || for custom")
            run = gr.Button("Generate", variant="primary")
        with gr.Column():
            audio_out = gr.Audio(label="Output", type="filepath", autoplay=True)

    language.change(_on_lang_change, inputs=language, outputs=text, show_progress=False)
    run.click(
        generate,
        inputs=[text, language, ref_audio, exaggeration, cfg_weight, temperature, pause_ms],
        outputs=audio_out,
    )


if __name__ == "__main__":
    # share=False keeps it local; set server_name="0.0.0.0" to expose on LAN.
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
