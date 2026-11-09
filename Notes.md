# Notes — getting natural-sounding speech

Practical hints for this project (Chatterbox Multilingual, EN / PL / RU).

---

## 1. Pauses & pacing

| Want | Do this |
|------|---------|
| Pause between sentences | Automatic. Set length with `--pause-ms 400` (CLI) or the **Pause** slider (UI). |
| A specific pause anywhere | Put `[pause:800]` in the text → 800 ms of silence. |
| A quick default pause | Put `||` in the text → uses the current pause setting. |
| Short breath / comma feel | Use punctuation: `,` short, `.` medium, `…` or `—` longer. |
| Slower overall delivery | Lower **CFG/pace** to ~0.3 and/or raise pause length. |

Example:
```
Добрый день. [pause:700] Сегодня || поговорим о погоде. Начнём.
```
Pauses are real silence inserted between separately-generated clips, so they're
exact and reliable (unlike punctuation, which only nudges prosody). Each clip's
own leading/trailing silence is auto-trimmed first, so the pause you set is the
real gap — e.g. **0 ms gives back-to-back sentences**. (Engine arg `trim_db=None`
keeps the model's natural tails if you prefer them.)

## 2. Keep chunks sentence-sized

The model is trained on short spans (~≤300 chars). Long paragraphs in a single
generation drift, rush, or hallucinate. This project auto-splits on `. ! ? …`
and newlines via `synthesize_long`, so **just write normal sentences** — each is
rendered separately and stitched. Break a very long sentence with a comma or a
`[pause:300]` if it still rushes.

## 3. Reference voice (biggest quality lever)

Without a reference clip, the default speaker sounds synthetic and (for RU/PL)
carries an English accent. Provide one:

- **5–15 seconds**, one speaker, clean (no music, no noise, no reverb).
- **Match the language** of your text (Russian clip for Russian).
- Mono WAV/FLAC is ideal; **.m4a/.mp3/.aac also work** (auto-transcoded via bundled FFmpeg).
- CLI: `--voice voices/ref.m4a` · UI: the *Reference voice* box.
- If the reference language ≠ text language on purpose (voice transfer), set
  **CFG/pace = 0** so the foreign accent isn't carried over.

## 4. Generation settings

| Setting | Natural default | Effect |
|---------|-----------------|--------|
| **Exaggeration** | `0.5` | Expressiveness. 0.3 = calm/flat, 0.7+ = dramatic but less stable. High values also speed speech up. |
| **CFG / pace** | `0.3–0.5` | Lower = slower, calmer, more human pacing. 1.0 rushes. `0` = cross-language voice transfer. |
| **Temperature** | `0.7–0.8` | Randomness. Low = steady but monotone; high (>1.2) = lively but risks wobble, slurring, hallucination. |

Good all-round starting point: **exaggeration 0.5, CFG 0.4, temperature 0.8**.
For expressive narration: **exaggeration 0.7 + CFG 0.3**.

## 5. Regenerate & keep the best

With **seed = 0** every run differs. If a take has an artifact, just regenerate
— generate 2–3 and keep the best. Found a great one? Fix the seed to reproduce it.

## 6. Russian stress (ударение)

Stress is auto-corrected via `ruaccent` (inserts the `U+0301` marks the model
needs). If a specific word is still stressed wrong, override it by typing the
combining acute accent right after the stressed vowel, e.g. `за́мок` vs `замо́к`.
Disable the whole feature with env var `CHATTERBOX_RU_ACCENT=0`.

## 7. Text normalization (read numbers/abbreviations right)

The model reads characters, not meaning. For reliable output, **spell things out**:
- Numbers: write `двадцать три` / `twenty-three`, not `23`.
- Dates/times/currency/units: expand (`$5` → `five dollars`, `10:30` → `half past ten`).
- Abbreviations: `т.е.` → `то есть`, `e.g.` → `for example`.
- Avoid ALL-CAPS (may be spelled letter-by-letter) unless you want an acronym spelled out.

## 8. Common pitfalls

- **Robotic / wrong accent** → add a native reference voice (see §3), not sliders.
- **Wrong word stress (RU)** → §6.
- **Rushed speech** → lower CFG (~0.3), raise pauses.
- **Glitches / repeats** → lower temperature (~0.7), or regenerate (§5), or shorten the sentence.
- **Mispronounced number/abbreviation** → spell it out (§7).
- **Max ~300 chars note** → advisory; `synthesize_long` handles longer text by chunking.
