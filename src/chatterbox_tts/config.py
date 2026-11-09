"""Project configuration.

Everything lives inside the project folder:
- Hugging Face model cache  -> <project>/.cache/huggingface
- Generated audio           -> <project>/outputs

Importing this module sets the HF cache environment variables *before*
chatterbox / transformers are imported, so no global user cache is touched.
"""
from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path


def _quiet_upstream_noise() -> None:
    """Hide harmless third-party warnings so real messages stand out."""
    # diffusers FutureWarning about LoRACompatibleLinear (not actionable here).
    warnings.filterwarnings("ignore", message=r".*LoRACompatibleLinear.*")
    warnings.filterwarnings("ignore", message=r".*torch.backends.cuda.sdp_kernel.*")
    # HF Hub "unauthenticated requests" notice (only affects download rate limits).
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)


if os.environ.get("CHATTERBOX_QUIET", "1") not in ("0", "false", "False"):
    _quiet_upstream_noise()

# <project> = two levels up from this file (src/chatterbox_tts/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
VOICES_DIR = PROJECT_ROOT / "voices"

# Keep all downloads inside the project. Must be set before importing HF libs.
os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "hub"))
# Faster, more robust downloads on Windows.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
# Windows without Developer Mode can't symlink the HF cache; silence the warning.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Languages this project is configured for (Chatterbox Multilingual codes).
# The model supports 23; we expose the two the project targets.
SUPPORTED = {
    "en": "English",
    "pl": "Polish",
    "ru": "Russian",
}


@dataclass
class Settings:
    """Runtime settings. Override via env vars or by editing defaults."""

    # "cuda" | "cpu" | "auto"
    device: str = os.environ.get("CHATTERBOX_DEVICE", "auto")
    # Default generation controls (see Chatterbox docs).
    exaggeration: float = float(os.environ.get("CHATTERBOX_EXAGGERATION", "0.5"))
    cfg_weight: float = float(os.environ.get("CHATTERBOX_CFG_WEIGHT", "0.5"))
    temperature: float = float(os.environ.get("CHATTERBOX_TEMPERATURE", "0.8"))
    # Max characters per generate() call (model is trained for short spans).
    max_chars: int = int(os.environ.get("CHATTERBOX_MAX_CHARS", "300"))

    cache_dir: Path = field(default=CACHE_DIR)
    output_dir: Path = field(default=OUTPUT_DIR)
    voices_dir: Path = field(default=VOICES_DIR)

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


settings = Settings()
