# Requirements

Local setup of **Chatterbox Multilingual TTS** (Resemble AI) configured for an
NVIDIA **RTX 5060 Laptop GPU** with CUDA, scoped to **English**, **Polish**, and **Russian**.
Everything (venv, model cache, outputs) lives inside this project folder.

---

## 1. Hardware

| Item   | This machine                          | Minimum / notes                                   |
|--------|---------------------------------------|---------------------------------------------------|
| GPU    | NVIDIA GeForce RTX 5060 Laptop (8 GB) | Blackwell, compute capability **sm_120**          |
| VRAM   | 8 GB                                   | Chatterbox 0.5B fits; keep batch/length modest    |
| RAM    | 32 GB                                  | 8 GB+ recommended                                 |
| Disk   | ~8–10 GB free                          | venv (~4 GB CUDA wheels) + model weights (~2–3 GB)|

## 2. System software

| Item           | Required                         | Verified on this machine         |
|----------------|----------------------------------|----------------------------------|
| OS             | Windows 10/11 64-bit             | Windows 11                       |
| NVIDIA driver  | ≥ 570 (exposes CUDA 12.8 / 13)   | **582.05** (CUDA 13 capable) ✅   |
| Python         | **3.10–3.12** (venv)             | **3.11.9** (in `.venv`) ✅        |
| Git            | any recent                       | ✅                                |
| Ollama         | optional, for the LLM bridge     | 0.40.9 ✅                         |

> ⚠️ The system default `python` is **3.14**, which PyTorch and Chatterbox do
> **not** support yet. The project venv is pinned to **Python 3.11**.
> A separate CUDA Toolkit install is **not** required — the PyTorch wheels bundle
> their own CUDA runtime. Only the NVIDIA driver matters.

## 3. The Blackwell / CUDA constraint (important)

`chatterbox-tts` pins **`torch==2.6.0`**, which has **no kernels for Blackwell
(sm_120)**. On an RTX 50-series card that build runs CPU-only or errors with
*"no kernel image is available for execution on the device"*.

**Fix applied in this project:** install PyTorch from the **CUDA 12.8** index
(`torch`/`torchaudio` ≥ 2.7, with `sm_120` support) and install Chatterbox
*without* its torch pin, so the cu128 build is kept.

```bash
# 1) CUDA 12.8 PyTorch (Blackwell-capable)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2) Chatterbox without deps, then its deps (torch stays at cu128)
pip install chatterbox-tts --no-deps
pip install "numpy>=1.24,<2.0" librosa==0.11.0 s3tokenizer transformers \
    diffusers conformer safetensors spacy-pkuseg pykakasi pyloudnorm omegaconf \
    "resemble-perth @ git+https://github.com/resemble-ai/Perth.git@master"
```

`pip` may print a dependency-conflict warning (Chatterbox *wants* torch 2.6.0);
this is expected and safe — the cu128 build is API-compatible.

## 4. Python dependencies

| Package                     | Version / source                 | Purpose                          |
|-----------------------------|----------------------------------|----------------------------------|
| torch, torchaudio           | ≥ 2.7, **cu128** index           | GPU tensor + audio I/O (Blackwell)|
| chatterbox-tts              | 0.1.x (deps unpinned here)       | the TTS model + pipeline         |
| transformers                | latest compatible                | Llama backbone / tokenizers      |
| diffusers                   | latest compatible                | flow / vocoder components        |
| librosa, pyloudnorm         | 0.11.0 / latest                  | audio processing                 |
| s3tokenizer, conformer      | latest                           | speech tokenizer / encoder       |
| safetensors, omegaconf      | latest                           | weights + config loading         |
| spacy-pkuseg, pykakasi      | latest                           | multilingual text handling       |
| resemble-perth              | git master                       | audio watermarking (built in)    |
| **requests** (extra)        | ≥ 2.31                           | *optional* Ollama bridge only    |

Install is reproducible from `requirements.lock.txt` (generated after setup) or
by re-running the steps above. A GPU-less machine can skip step 1 and just
`pip install chatterbox-tts`.

## 5. Language support

Configured codes (Chatterbox Multilingual supports 23 total):

| Code | Language |
|------|----------|
| `en` | English  |
| `pl` | Polish   |
| `ru` | Russian  |

## 6. Ollama (optional)

Ollama is **not needed for TTS**. It is included only as an optional bridge:
an LLM generates text, Chatterbox speaks it. See
`src/chatterbox_tts/ollama_helper.py`. Enable with:

```bash
pip install -e .[ollama]
```

Local models available: `qwen3.5:9b`, `llama3.1:8b`, `llama3.2:1b`,
`qwen2.5vl:7b`, `mxbai-embed-large`.

## 7. Verify

```bash
python examples/check_gpu.py                      # expects cuda available: True, sm_120
python examples/synthesize.py --demo              # writes outputs/demo_en.wav, demo_pl.wav
```
