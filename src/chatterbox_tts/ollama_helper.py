"""Optional Ollama helper.

Chatterbox does NOT need Ollama. This module is a small, opt-in bridge so you
can later feed LLM-generated text into the TTS engine (e.g. "say the LLM's
answer out loud"). It talks to a local Ollama server over HTTP; nothing here is
imported by the core TTS path.

Requires the `ollama` server running (default http://localhost:11434) and the
`requests` package (installed as an extra, not a core TTS dependency).

Example
-------
>>> from chatterbox_tts.ollama_helper import generate_text
>>> txt = generate_text("Przywitaj sie po polsku w jednym zdaniu.", model="llama3.1:8b")
>>> from chatterbox_tts import TTSEngine
>>> TTSEngine().synthesize(txt, language="pl", out="outputs/llm_pl.wav")
"""
from __future__ import annotations

import os

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")


def generate_text(prompt: str, model: str = DEFAULT_MODEL, host: str = OLLAMA_HOST) -> str:
    """Return a single completion string from a local Ollama model."""
    import requests

    resp = requests.post(
        f"{host}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def list_models(host: str = OLLAMA_HOST) -> list[str]:
    import requests

    resp = requests.get(f"{host}/api/tags", timeout=30)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]
