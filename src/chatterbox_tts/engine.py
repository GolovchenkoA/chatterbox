"""Thin, GPU-aware wrapper around ChatterboxMultilingualTTS.

Example
-------
>>> from chatterbox_tts import TTSEngine
>>> eng = TTSEngine()            # loads model onto CUDA if available
>>> eng.synthesize("Hello world", language="en", out="outputs/hello.wav")
>>> eng.synthesize("Dzien dobry", language="pl", out="outputs/pl.wav")
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

# Importing config first sets HF_HOME before transformers/chatterbox load.
from .config import settings, SUPPORTED


class TTSEngine:
    def __init__(self, device: Optional[str] = None):
        import torch  # local import so config env vars are set first

        self.device = device or settings.resolve_device()
        if self.device == "cuda" and not torch.cuda.is_available():
            print("[chatterbox] CUDA requested but not available -> falling back to CPU")
            self.device = "cpu"

        self._log_device(torch)

        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        print(f"[chatterbox] loading model to {self.device} (first run downloads weights)...")
        self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        self.sr = self.model.sr
        print("[chatterbox] model ready.")

    @staticmethod
    def _log_device(torch) -> None:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = ".".join(map(str, torch.cuda.get_device_capability(0)))
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(
                f"[chatterbox] CUDA ok: {name} (sm_{cap.replace('.', '')}, "
                f"{total:.1f} GB) | torch {torch.__version__} | cuda {torch.version.cuda}"
            )
        else:
            print(f"[chatterbox] CUDA not available | torch {torch.__version__}")

    def synthesize(
        self,
        text: str,
        language: str = "en",
        out: str | Path | None = None,
        audio_prompt_path: str | Path | None = None,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
        temperature: float | None = None,
    ) -> Path:
        """Generate speech for `text` in `language` and save a WAV file.

        `audio_prompt_path` (optional) is a reference clip to clone the voice;
        it should match the target language for best results.
        """
        import soundfile as sf

        if language not in SUPPORTED:
            raise ValueError(
                f"Language '{language}' not configured. Options: {list(SUPPORTED)}"
            )

        if len(text) > settings.max_chars:
            print(
                f"[chatterbox] text > {settings.max_chars} chars; "
                "consider splitting into sentences for best quality."
            )

        kwargs = dict(
            language_id=language,
            exaggeration=exaggeration if exaggeration is not None else settings.exaggeration,
            cfg_weight=cfg_weight if cfg_weight is not None else settings.cfg_weight,
            temperature=temperature if temperature is not None else settings.temperature,
        )
        if audio_prompt_path:
            kwargs["audio_prompt_path"] = str(audio_prompt_path)

        wav = self.model.generate(text, **kwargs)

        if out is None:
            out = settings.output_dir / f"{language}_output.wav"
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        # soundfile avoids torchaudio 2.11's torchcodec/FFmpeg backend dependency.
        audio = wav.detach().cpu().numpy().squeeze()  # [1, N] -> [N]
        sf.write(str(out), audio, self.sr)
        print(f"[chatterbox] saved -> {out}")
        return out


# Module-level convenience (loads the model each call; prefer TTSEngine for reuse).
def synthesize(text: str, language: str = "en", out: str | Path | None = None, **kw) -> Path:
    return TTSEngine().synthesize(text, language=language, out=out, **kw)
