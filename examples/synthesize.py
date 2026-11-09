"""CLI: synthesize speech with Chatterbox Multilingual (EN / PL).

Usage
-----
    python examples/synthesize.py --lang en --text "Hello from my GPU."
    python examples/synthesize.py --lang pl --text "Dzien dobry, to jest test."
    python examples/synthesize.py --demo         # generate one EN + one PL sample
    python examples/synthesize.py --lang pl --text "..." --voice voices/ref.wav
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable when run directly, without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chatterbox_tts import TTSEngine, settings  # noqa: E402

DEMO = {
    "en": "Last month we reached two billion views on our YouTube channel.",
    "pl": "W zeszlym miesiacu osiagnelismy dwa miliardy wyswietlen na naszym kanale YouTube.",
    "ru": "В прошлом месяце мы достигли нового рубежа: два миллиарда просмотров на нашем YouTube-канале.",
}


def main() -> None:
    p = argparse.ArgumentParser(description="Chatterbox Multilingual TTS (EN/PL)")
    p.add_argument("--lang", choices=["en", "pl", "ru"], default="en")
    p.add_argument("--text", help="Text to synthesize")
    p.add_argument("--out", help="Output .wav path")
    p.add_argument("--voice", help="Reference audio clip to clone (optional)")
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.add_argument("--exaggeration", type=float, default=None)
    p.add_argument("--cfg-weight", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--pause-ms", type=int, default=400,
                   help="Silence between sentences (ms). Use [pause:N] or || in text for custom gaps.")
    p.add_argument("--demo", action="store_true", help="Generate one EN and one PL sample")
    args = p.parse_args()

    engine = TTSEngine(device=args.device)

    if args.demo:
        for lang, text in DEMO.items():
            engine.synthesize(
                text, language=lang, out=settings.output_dir / f"demo_{lang}.wav"
            )
        return

    if not args.text:
        p.error("provide --text or use --demo")

    engine.synthesize_long(
        args.text,
        language=args.lang,
        out=args.out,
        pause_ms=args.pause_ms,
        audio_prompt_path=args.voice,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
