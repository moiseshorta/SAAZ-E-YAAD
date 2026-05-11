# Saaz-e-Yaad

An autoregressive music generation system based on Continuous Autoregressive Models
(CAM) with CLAP conditioning, operating over `music2latent` audio latents. The
repository includes the training and inference pipeline plus an Electron desktop
application for interactive generation.

## Background

Implementation of:

> *Continuous Autoregressive Models with Noise Augmentation Avoid Error Accumulation*
> Pasini, Nistal, Lattner, Fazekas — NeurIPS 2024 Audio Imagination Workshop
> [arXiv:2411.18447](https://arxiv.org/abs/2411.18447)

The model is an autoregressive Transformer backbone paired with a small per-frame
sampler MLP that performs a diffusion-style denoising loop. Training injects noise
into the latent context so the model is robust to its own prediction drift at
inference time. Conditioning is supplied as a sequence of CLAP tokens (from audio
or text) consumed via cross-attention.

## Repository layout

```
.
├── preprocess.py      # Encode audio into music2latent + CLAP tokens (.pt files)
├── train.py           # Train the CAM model (CFG dropout, EMA, AMP, warmup+cosine)
├── generate.py        # Inference with CFG, optional audio/text conditioning and prefix
├── export.py          # Export a trained checkpoint to a TorchScript module
├── generate-ts.py     # Run inference directly from an exported TorchScript model
├── requirements.txt
└── electron-gui/      # Desktop application (see below)
```

The Python scripts are self-contained — model code lives inside `train.py` and is
loaded by `generate.py` / `export.py` from the saved checkpoint.

## Requirements

- Python 3.10+
- PyTorch 2.2+ (CUDA, MPS or CPU)
- `music2latent` (from source)
- See `requirements.txt` for the full list

Install in a fresh environment:

```bash
conda create -n cam python=3.10 -y
conda activate cam
pip install -r requirements.txt
```

## Pipeline

### 1. Preprocess

Encode a directory of audio files into per-track `.pt` files containing
`music2latent` frames and CLAP tokens:

```bash
python preprocess.py \
  --audio_dir /path/to/audio \
  --output_dir preprocessed \
  --sample_rate 44100 \
  --clap_window_seconds 5.0 \
  --clap_hop_seconds 5.0
```

### 2. Train

```bash
python train.py \
  --preprocessed_dir preprocessed \
  --checkpoint_dir checkpoints \
  --batch_size 32 \
  --lr 1e-4 \
  --warmup_steps 4000
```

Training uses AdamW with linear warmup followed by cosine decay, EMA weights,
mixed precision (with MPS-safe fallback), classifier-free guidance dropout, and
periodic validation samples.

### 3. Generate

```bash
python generate.py \
  --checkpoint checkpoints/checkpoint_50000.pt \
  --output_dir samples \
  --num_samples 4 \
  --gen_seconds 30 \
  --cfg_scale 5.0 \
  --temperature 1.0 \
  --diffusion_steps 24
```

Supports three conditioning modes: unconditional, text prompt, or audio prompt
(multi-token CLAP). An optional prefix audio file can seed the first N latent
frames for continuation.

### 4. Export to TorchScript

```bash
python export.py \
  --checkpoint checkpoints/checkpoint_50000.pt \
  --output ts-models/model.ts
```

The exported `.ts` file is what the Electron app loads at runtime.

### 5. Inference from TorchScript

```bash
python generate-ts.py \
  --ts_model ts-models/model.ts \
  --output_dir samples
```

## Electron application

`electron-gui/` contains a desktop app that wraps the TorchScript inference path.
The main process spawns a Python backend (`backend.py`) and communicates with it
via line-delimited JSON over stdio. The renderer provides model selection,
audio/text conditioning, prefix continuation, streaming preview playback and a
waveform display.

### Development

```bash
cd electron-gui
npm install
npm start
```

In development mode the renderer launches the Python backend directly. Set
`CAM_PYTHON` to point at the desired interpreter if the default is not the one
with your dependencies installed.

### Distribution build

```bash
cd electron-gui
npm run dist
```

This runs three steps:

1. `prepare-build.mjs` — checks `build/icon.icns` and other assets.
2. `build-backend.mjs` — runs PyInstaller against `backend.spec` to produce a
   standalone backend executable. The script picks the Python interpreter from
   `CAM_PYTHON`, then `CONDA_PREFIX/bin/python`, then `PYTHON`, then the system
   `python`/`python3`. It performs a preflight import check on `librosa`,
   `music2latent`, `transformers` and `soundfile` to avoid producing a binary
   that fails at runtime.
3. `electron-builder` — packages the renderer, main process and the bundled
   backend into a macOS `.dmg` / `.zip` (other targets configurable in
   `package.json`).

TorchScript models placed in `../ts-models/` are bundled into the resulting app
under `Contents/Resources/ts-models/`.

## License

Code is provided as-is for research use. See the original paper for the method.
