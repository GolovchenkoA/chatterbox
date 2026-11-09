"""Reference-audio loading.

`soundfile` (libsndfile) reads WAV/FLAC/OGG but NOT compressed formats like
.m4a/.aac/.mp3. To accept those as voice references, we transcode them to WAV
using a **bundled** FFmpeg binary from `imageio-ffmpeg` — no system FFmpeg
install required. Converted files are cached under <project>/.cache/ref_audio.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from .config import settings

# Formats soundfile/libsndfile can read directly (no transcode needed).
NATIVE_EXTS = {".wav", ".flac", ".ogg", ".oga", ".aif", ".aiff", ".w64"}

_REF_DIR = settings.cache_dir / "ref_audio"


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as e:
        raise RuntimeError(
            "Reading .m4a/.mp3/.aac references needs a transcoder. Install it with:\n"
            "    pip install imageio-ffmpeg\n"
            "(or supply the reference as .wav/.flac/.ogg instead)."
        ) from e


def ensure_wav(path: str | Path, target_sr: int | None = None) -> str:
    """Return a path to a WAV usable as a reference clip.

    Native formats pass through unchanged. Compressed/other formats are
    transcoded to mono WAV (optionally at `target_sr`) via bundled FFmpeg and
    cached, so re-using the same file is instant.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix.lower() in NATIVE_EXTS:
        import soundfile as sf

        try:
            sf.info(str(p))  # confirm libsndfile can actually read it
            return str(p)
        except Exception:
            pass  # fall through to transcode

    _REF_DIR.mkdir(parents=True, exist_ok=True)
    # Cache key from path + size + mtime so edits re-convert.
    stat = p.stat()
    key = hashlib.sha1(f"{p.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{target_sr}".encode()).hexdigest()[:16]
    out = _REF_DIR / f"{p.stem}_{key}.wav"
    if out.exists():
        return str(out)

    cmd = [_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(p), "-ac", "1"]
    if target_sr:
        cmd += ["-ar", str(target_sr)]
    cmd += [str(out)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to transcode '{p.name}' to WAV: {e.stderr.decode(errors='ignore')[:300]}"
        ) from e
    print(f"[chatterbox] transcoded reference {p.suffix} -> {out.name}")
    return str(out)
