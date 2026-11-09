"""Russian stress (ударение) preprocessing.

Chatterbox's Russian expects stress-marked text (the model vocab contains the
combining acute accent U+0301). The package it normally uses for this
(`russian_text_stresser`) is unavailable, so we generate the marks with
`ruaccent` and convert them to the U+0301 form the model was trained on.

`ruaccent.process_all` marks stress with a '+' placed BEFORE the stressed vowel
(e.g. "прив+ет"). We rewrite that to the vowel followed by U+0301
(e.g. "приве́т"), which is what Chatterbox's tokenizer expects.

Enable/disable via env var CHATTERBOX_RU_ACCENT (default "1").
Install with: pip install ruaccent   (or: pip install -e .[ru])
"""
from __future__ import annotations

import os
import re

ACUTE = "́"  # combining acute accent — the model's stress token
_VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
# "+<vowel>"  ->  "<vowel>U+0301"
_PLUS_VOWEL = re.compile(r"\+([" + _VOWELS + r"])")

_accentizer = None  # lazy singleton


def _plus_to_acute(text: str) -> str:
    text = _PLUS_VOWEL.sub(lambda m: m.group(1) + ACUTE, text)
    return text.replace("+", "")  # drop any stray '+' not before a vowel


def _get_accentizer():
    global _accentizer
    if _accentizer is None:
        from ruaccent import RUAccent

        acc = RUAccent()
        # 'turbo3.1' = good omograph model; dictionary for known words.
        # Model downloads once into HF_HOME (project .cache).
        acc.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)
        _accentizer = acc
    return _accentizer


def add_stress(text: str) -> str:
    """Return `text` with Russian lexical stress marked using U+0301.

    Falls back to the original text (with a printed note) if ruaccent is not
    installed, so synthesis still works — just without corrected stress.
    """
    if os.environ.get("CHATTERBOX_RU_ACCENT", "1") not in ("1", "true", "True"):
        return text
    try:
        plus_marked = _get_accentizer().process_all(text)
        return _plus_to_acute(plus_marked)
    except ImportError:
        print(
            "[chatterbox] ruaccent not installed -> Russian stress not corrected. "
            "Install with: pip install ruaccent"
        )
        return text
    except Exception as e:  # never block synthesis on stressing errors
        print(f"[chatterbox] Russian stressing failed ({e}); using unmarked text.")
        return text
