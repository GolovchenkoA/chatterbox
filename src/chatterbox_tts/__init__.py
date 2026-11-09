"""Local Chatterbox Multilingual TTS project package.

Public API:
    from chatterbox_tts import TTSEngine, synthesize
"""
from .engine import TTSEngine, synthesize
from .config import SUPPORTED, settings

__all__ = ["TTSEngine", "synthesize", "SUPPORTED", "settings"]
