"""Thin, GPU-aware wrapper around ChatterboxMultilingualTTS.

Example
-------
>>> from chatterbox_tts import TTSEngine
>>> eng = TTSEngine()            # loads model onto CUDA if available
>>> eng.synthesize("Hello world", language="en", out="outputs/hello.wav")
>>> eng.synthesize("Dzien dobry", language="pl", out="outputs/pl.wav")
"""
from __future__ import annotations

import re
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

        print(f"[chatterbox] loading model to {self.device} (downloads weights on first run only)...")
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
        if language not in SUPPORTED:
            raise ValueError(
                f"Language '{language}' not configured. Options: {list(SUPPORTED)}"
            )
        if len(text) > settings.max_chars:
            print(
                f"[chatterbox] text > {settings.max_chars} chars; "
                "use synthesize_long() to split into sentences for best quality."
            )
        gen = self._gen_kwargs(language, audio_prompt_path, exaggeration, cfg_weight, temperature)
        audio = self._render(text, language, gen)
        return self._save(audio, out or settings.output_dir / f"{language}_output.wav")

    def synthesize_long(
        self,
        text: str,
        language: str = "en",
        out: str | Path | None = None,
        pause_ms: int = 400,
        sentence_pause_ms: int | None = None,
        audio_prompt_path: str | Path | None = None,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
        temperature: float | None = None,
        trim_db: float | None = 25.0,
        max_words: int | None = 15,
    ) -> Path:
        """Synthesize arbitrarily long text with controllable pauses.

        - Splits text into sentences and inserts `sentence_pause_ms` of silence
          between them (defaults to `pause_ms`). Also avoids the ~300-char limit.
        - `max_words`: pack sentences into chunks of up to this many words per
          generate() call (None/0 = one sentence per chunk). Larger = fewer,
          longer clips = faster overall, but pauses only fall between chunks.
        - Insert a custom pause anywhere with a marker:
            `[pause:800]`  -> 800 ms of silence
            `||`           -> `pause_ms` of silence
        - `trim_db`: trim each clip's leading/trailing silence at this threshold
          so the *inserted* pause is the real gap (the model pads clips with
          ~0.5-1 s of silence otherwise). Set None to disable trimming.
        """
        import numpy as np

        if language not in SUPPORTED:
            raise ValueError(
                f"Language '{language}' not configured. Options: {list(SUPPORTED)}"
            )
        sent_pause = pause_ms if sentence_pause_ms is None else sentence_pause_ms
        gen = self._gen_kwargs(language, audio_prompt_path, exaggeration, cfg_weight, temperature)

        pieces: list = []
        for kind, val in split_segments(text, pause_ms=pause_ms, sentence_pause_ms=sent_pause,
                                        max_words=max_words or None):
            if kind == "pause":
                pieces.append(self._silence(val))
            else:
                print(f"[chatterbox] rendering: '{val[:60]}'")
                pieces.append(self._trim(self._render(val, language, gen), trim_db))

        audio = np.concatenate(pieces) if pieces else np.zeros(1, dtype="float32")
        return self._save(audio, out or settings.output_dir / f"{language}_long.wav")

    # --- internals -------------------------------------------------------
    def _gen_kwargs(self, language, audio_prompt_path, exaggeration, cfg_weight, temperature):
        kwargs = dict(
            language_id=language,
            exaggeration=exaggeration if exaggeration is not None else settings.exaggeration,
            cfg_weight=cfg_weight if cfg_weight is not None else settings.cfg_weight,
            temperature=temperature if temperature is not None else settings.temperature,
        )
        if audio_prompt_path:
            from .audio_io import ensure_wav

            # Accept .m4a/.mp3/etc. by transcoding to WAV (soundfile can't read them).
            kwargs["audio_prompt_path"] = ensure_wav(audio_prompt_path)
        return kwargs

    def _render(self, text, language, gen_kwargs):
        """Generate one chunk and return a 1-D float32 numpy waveform."""
        if language == "ru":
            from .ru_stress import add_stress

            text = add_stress(text)
        wav = self.model.generate(text, **gen_kwargs)
        return wav.detach().cpu().numpy().astype("float32").squeeze()

    def _trim(self, audio, trim_db):
        """Trim leading/trailing silence so inserted pauses are the real gap.

        Keeps a small guard (~15 ms) at each edge so word onsets aren't clipped.
        """
        if trim_db is None:
            return audio
        import librosa
        import numpy as np

        _, idx = librosa.effects.trim(audio, top_db=trim_db)
        guard = int(self.sr * 0.015)
        start = max(0, idx[0] - guard)
        end = min(len(audio), idx[1] + guard)
        out = audio[start:end]
        return out if out.size else audio

    def _silence(self, ms: int):
        import numpy as np

        return np.zeros(int(self.sr * max(0, ms) / 1000), dtype="float32")

    def _save(self, audio, out):
        import soundfile as sf

        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        # soundfile avoids torchaudio 2.11's torchcodec/FFmpeg backend dependency.
        sf.write(str(out), audio, self.sr)
        print(f"[chatterbox] saved -> {out}")
        return out


# Sentence splitter: break after . ! ? … (keeping the punctuation) and on newlines.
_SENT_SPLIT = re.compile(r"(?<=[.!?…。！？])\s+|\n+")
# Explicit pause markers: [pause:NUMBER] or ||
_PAUSE_MARK = re.compile(r"(\[pause:(\d+)\]|\|\|)")


def _pack_words(sentences: list[str], max_words: int) -> list[str]:
    """Greedily group sentences into chunks of at most `max_words` words.

    A single sentence longer than `max_words` is split at word boundaries.
    """
    chunks: list[str] = []
    cur: list[str] = []
    cur_w = 0
    for s in sentences:
        words = s.split()
        if len(words) > max_words:
            if cur:
                chunks.append(" ".join(cur))
                cur, cur_w = [], 0
            for j in range(0, len(words), max_words):
                chunks.append(" ".join(words[j:j + max_words]))
            continue
        if cur and cur_w + len(words) > max_words:
            chunks.append(" ".join(cur))
            cur, cur_w = [], 0
        cur.append(s)
        cur_w += len(words)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def split_segments(text: str, pause_ms: int = 400, sentence_pause_ms: int = 400,
                   max_words: int | None = None):
    """Yield ('speak', chunk) / ('pause', ms) segments from text.

    Explicit markers ([pause:N], ||) produce pauses and are always hard chunk
    boundaries. Everything else is split into sentences; if `max_words` is set,
    sentences are packed into chunks of up to that many words (fewer, larger
    generate() calls). A `sentence_pause_ms` gap is inserted between chunks.
    """
    segments: list[tuple[str, object]] = []
    for part in _PAUSE_MARK.split(text):
        if part is None or part == "":
            continue
        if part == "||":
            segments.append(("pause", pause_ms))
            continue
        m = re.fullmatch(r"\[pause:(\d+)\]", part)
        if m:
            segments.append(("pause", int(m.group(1))))
            continue
        if part.isdigit():  # the capture group of the ms number — skip it
            continue
        sentences = [s.strip() for s in _SENT_SPLIT.split(part) if s.strip()]
        chunks = _pack_words(sentences, max_words) if max_words else sentences
        for i, c in enumerate(chunks):
            if i > 0:
                segments.append(("pause", sentence_pause_ms))
            segments.append(("speak", c))
    return segments


# Module-level convenience (loads the model each call; prefer TTSEngine for reuse).
def synthesize(text: str, language: str = "en", out: str | Path | None = None, **kw) -> Path:
    return TTSEngine().synthesize(text, language=language, out=out, **kw)
