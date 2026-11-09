# Chatterbox Multilingual TTS — Local (EN / PL, CUDA)

Local, GPU-accelerated text-to-speech using
[Resemble AI Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox),
configured for an **NVIDIA RTX 5060 Laptop GPU** (Blackwell, CUDA 12.8) and scoped
to **English** and **Polish**. Everything — virtualenv, model cache, and output
audio — stays inside this project folder.

See [`Requirements.md`](Requirements.md) for the full hardware/software matrix and
the Blackwell CUDA constraint.

> ## ⚠️ Required: Smart App Control must be OFF
> Chatterbox depends on `numba`/`llvmlite`, whose unsigned native DLLs are blocked
> by Windows **Smart App Control** (symptom: `ImportError: DLL load failed while
> importing _dynfunc` / *"Zasady kontroli aplikacji zablokowały ten plik"*).
> Nothing — CLI or UI — will import the model until it is turned off.
>
> **Turn it off:** Windows Security → *App & browser control* → *Smart App Control
> settings* → **Off**, then reboot.
> (PL: *Zabezpieczenia Windows → Kontrola aplikacji i przeglądarki → Ustawienia
> inteligentnego sterowania aplikacjami → Wyłącz*, potem restart.)
>
> ⚠️ Once Off, Smart App Control **cannot be turned back on** without resetting/
> reinstalling Windows. This is a one-way security trade-off.

## Setup (already done in this project)

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate

# Blackwell-capable PyTorch (CUDA 12.8)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Chatterbox without its torch pin, then deps + this package
pip install --no-deps "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git"
pip install "numpy>=1.24,<2.0" librosa==0.11.0 s3tokenizer transformers==5.2.0 \
    diffusers==0.29.0 conformer==0.3.2 safetensors==0.5.3 spacy-pkuseg \
    pykakasi==2.3.0 pyloudnorm omegaconf \
    "resemble-perth @ git+https://github.com/resemble-ai/Perth.git@master"
pip install -e . --no-deps
```

### Reproduce from the lockfile (fresh machine)

`requirements.lock.txt` pins every package, so install with `--no-deps`
(otherwise pip tries to resolve Chatterbox's `torch==2.6.0` pin and fails):

```bash
pip install -r requirements.lock.txt --no-deps \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple
pip install -e . --no-deps
```

## Usage

```bash
.venv\Scripts\activate

# Check the GPU is used
python examples/check_gpu.py

# One English + one Polish sample -> outputs/
python examples/synthesize.py --demo

# Custom text
python examples/synthesize.py --lang en --text "Hello from my GPU."
python examples/synthesize.py --lang pl --text "Dzien dobry, to jest test."

# Clone a voice from a reference clip (match the language)
python examples/synthesize.py --lang pl --text "..." --voice voices/ref.wav
```

### Web UI (no code)

A local Gradio browser interface (reuses the same GPU engine, EN/PL):

```bash
.venv\Scripts\activate
python app_ui.py
```

Opens http://127.0.0.1:7860 — enter text, pick English/Polish, optionally upload a
reference voice, click **Generate**. Output also saved to `outputs/`.
(gradio is installed; the `ui` extra `pip install -e .[ui]` pins it too.)

From Python:

```python
from chatterbox_tts import TTSEngine
eng = TTSEngine()                      # loads once onto CUDA
eng.synthesize("Hello world", "en", "outputs/hi.wav")
eng.synthesize("Dzien dobry", "pl", "outputs/pl.wav")
```

## Optional: Ollama bridge

TTS does not need Ollama. To have a local LLM generate text that Chatterbox then
speaks:

```bash
pip install -e .[ollama]
```

```python
from chatterbox_tts.ollama_helper import generate_text
from chatterbox_tts import TTSEngine
text = generate_text("Napisz jedno zdanie powitania.", model="llama3.1:8b")
TTSEngine().synthesize(text, "pl", "outputs/llm_pl.wav")
```

## Project layout

```
.
├── Requirements.md            # full requirements + CUDA/Blackwell notes
├── README.md
├── pyproject.toml             # local package (torch installed separately)
├── src/chatterbox_tts/
│   ├── config.py              # project-local cache/output, device selection
│   ├── engine.py              # TTSEngine wrapper (CUDA-aware)
│   └── ollama_helper.py       # optional LLM bridge
├── examples/
│   ├── check_gpu.py
│   └── synthesize.py          # CLI
├── outputs/                   # generated audio (gitignored)
└── .cache/huggingface/        # model weights (gitignored)
```

## Notes
- First run downloads model weights (~2–3 GB) into `.cache/huggingface/`.
- Keep inputs ≤ ~300 characters per call; split longer text into sentences.
- All generated audio carries Resemble's inaudible **PerTh watermark**.
- Model + library are **MIT licensed**.
