# SONIC SHIKARA - Continuous Autoregressive Music Generation

![SONIC SHIKARA](https://img.shields.io/badge/SONIC-SHIKARA-blue)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.2+-orange)
![Electron](https://img.shields.io/badge/electron-28.2+-purple)

An implementation of **Continuous Autoregressive Models with Noise Augmentation** for music generation, featuring a modern Electron GUI and state-of-the-art v4-xl architecture with CLAP conditioning.

## 🎵 Overview

SONIC SHIKARA implements the paper:

> **Continuous Autoregressive Models with Noise Augmentation Avoid Error Accumulation**  
> [Marco Pasini, Javier Nistal, Stefan Lattner, György Fazekas]  
> NeurIPS 2024, Audio Imagination Workshop  
> [[Paper on arXiv]](https://arxiv.org/abs/2411.18447)

### Key Features

- **Hybrid AR + Diffusion Architecture**: Backbone provides autoregressive context, sampler MLP does per-frame diffusion
- **CLAP Conditioning**: Multi-modal text/audio conditioning with interleaved cross-attention
- **KV-Cached Generation**: O(T) per-step autoregressive generation for efficient inference
- **Electron GUI**: Cross-platform desktop application with real-time streaming preview
- **TorchScript Export**: Optimized models for deployment in the packaged app
- **Multiple Model Variants**: Full, rhythm, harmony, and vocal models
- **SOTA Techniques**: Logit-normal noise schedule, Min-SNR-γ loss weighting, RoPE embeddings, flash attention

## 🏗️ Architecture

### CAM v4-xl Model

The v4-xl architecture includes:

- **Backbone Transformer**: 24-layer causal self-attention with RoPE positional embeddings
- **Interleaved Cross-Attention**: CLAP conditioning injected every 4 backbone layers
- **Sampler MLP**: 12-layer AdaLayerNorm MLP with midpoint cross-attention to CLAP
- **KV-Cache Support**: Efficient O(T) per-step generation
- **Mixed Precision Training**: AMP with MPS safety for Apple Silicon
- **CFG with Rescaling**: Classifier-free guidance with optional rescaling

### Electron GUI

- **Main Process** (`main.js`): Electron main process, backend process management, file dialogs
- **Renderer** (`renderer.js`): UI logic, streaming audio preview, waveform visualization
- **Backend** (`backend.py`): Python generation server with JSON-over-stdout protocol
- **Communication**: IPC between main/renderer, stdin/stdout JSON protocol for backend

## 📦 Installation

### Prerequisites

- Python 3.10+
- Conda (recommended) or virtual environment
- Node.js 18+ (for Electron GUI build)
- PyInstaller (for backend packaging)

### Python Environment Setup

```bash
# Create conda environment
conda create -n cam python=3.10
conda activate cam

# Install dependencies
pip install torch>=2.2
pip install transformers>=4.38
pip install numpy>=1.24
pip install scipy>=1.10
pip install librosa>=0.10.0
pip install soundfile>=0.12.1
pip install tqdm>=4.66
pip install rich>=13.7
pip install gradio>=4.0
pip install music2latent @ git+https://github.com/moiseshorta/music2latent.git
```

### Electron GUI Development

```bash
cd electron-gui
npm install
npm start  # Run in development mode
```

## 🚀 Usage

### CLI Training Pipeline

#### 1. Preprocess Audio Data

```bash
python preprocess-clap-v4-xl.py \
  --audio_dir /path/to/audio \
  --output_dir preprocessed_v4 \
  --sample_rate 48000 \
  --clap_window_seconds 5.0 \
  --clap_hop_seconds 5.0
```

#### 2. Train Model

```bash
python train-clap-v4-warmup-xl-cfg.py \
  --preprocessed_dir preprocessed_v4 \
  --checkpoint_dir checkpoints_v4 \
  --batch_size 32 \
  --lr 1e-4 \
  --warmup_steps 4000 \
  --max_steps 100000 \
  --validation_every 1000 \
  --save_every 5000
```

#### 3. Generate Audio

```bash
python generate-clap-v4-xl-warmup-val-cfg.py \
  --checkpoint checkpoints_v4/checkpoint_50000.pt \
  --output_dir inference_samples_v4 \
  --num_samples 4 \
  --gen_seconds 30.5 \
  --cfg_scale 5.0 \
  --temperature 1.0 \
  --diffusion_steps 24
```

#### 4. Export to TorchScript

```bash
python export-clap-v4-xl.py \
  --checkpoint checkpoints_v4/checkpoint_50000.pt \
  --output ts-models/sonic-shikara_full.ts
```

### Electron GUI

#### Development Mode

```bash
cd electron-gui
npm start
```

#### Build for Distribution

```bash
cd electron-gui
npm run build:prepare  # Prepare build assets
npm run build:backend  # Build Python backend with PyInstaller
npm run dist           # Build Electron app (macOS DMG/ZIP)
```

The build process:
1. Packages the Python backend using PyInstaller with dependency checks
2. Bundles TorchScript models from `ts-models/`
3. Creates macOS app with code signing and notarization
4. Outputs to `dist/` directory

#### GUI Features

- **Model Selection**: Choose from bundled or custom TorchScript models
- **Audio Conditioning**: Drag-and-drop audio for CLAP conditioning
- **Prefix Continuation**: Continue from existing audio with configurable duration
- **Real-time Preview**: Streaming audio playback during generation
- **Waveform Visualization**: Live waveform display
- **Parameter Controls**: CFG scale, temperature, diffusion steps, generation length
- **Output Management**: Browse and play generated samples

## 🎯 Model Variants

The project includes several pre-trained models:

- **sonic-shikara_full.ts**: Full music generation model
- **sonic-shikara_rhythm.ts**: Rhythm-focused generation
- **sonic-shikara_rhythm-v4.ts**: Rhythm model with v4 architecture
- **sonic-shikara_harmony.ts**: Harmony-focused generation
- **sonic-shikara_vocals.ts**: Vocal-focused generation

## 🔧 Configuration

### Training Parameters

Key training hyperparameters (see `train-clap-v4-warmup-xl-cfg.py`):

- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-4)
- `--warmup_steps`: Linear warmup steps (default: 4000)
- `--weight_decay`: AdamW weight decay (default: 0.01)
- `--cfg_drop_prob`: CFG dropout probability (default: 0.1)
- `--snr_gamma`: Min-SNR-γ loss weighting (default: 5.0)
- `--ema_decay`: EMA decay rate (default: 0.999)

### Generation Parameters

Key generation parameters (see `generate-clap-v4-xl-warmup-val-cfg.py`):

- `--cfg_scale`: Classifier-free guidance scale (default: 5.0)
- `--temperature`: Sampling temperature (default: 1.0, max: 1.25)
- `--diffusion_steps`: Number of diffusion steps (default: 24, max: 50)
- `--gen_seconds`: Generation length in seconds (default: 30.5)
- `--solver`: ODE solver - midpoint or euler (default: midpoint)

## 📊 Technical Details

### Noise Augmentation

CAM injects noise during training to simulate error accumulation:

1. **Backbone Noise**: `x_tilde = k*ε + (1-k)*x` with uniform k ∈ [0,1]
2. **Sampler Noise**: Logit-normal noise schedule with tunable mean/std
3. **History Dropout**: Randomly drops context to zSOS (20% probability)

### Rectified Flow

The sampler uses rectified flow targets:
- Target: `v = x - ε`
- Loss: MSE on velocity prediction
- Optional CFG rescaling to prevent oversaturation

### SOTA Techniques Implemented

- ✅ Logit-normal timestep sampling
- ✅ Min-SNR-γ loss weighting
- ✅ RoPE positional embeddings (correct ordering)
- ✅ Flash attention (F.scaled_dot_product_attention)
- ✅ KV-cache for AR generation
- ✅ Batched cond/uncond CFG passes
- ✅ Midpoint (Heun) 2nd-order ODE solver
- ✅ Gradient checkpointing
- ✅ EMA weights
- ✅ Cosine LR schedule + linear warmup
- ✅ Mixed precision (AMP)

See `V4_XL_EVALUATION.md` for detailed SOTA alignment analysis.

## 🛠️ Build Requirements

### macOS Build

The Electron app includes macOS-specific build configuration:

- **Code Signing**: Hardened runtime with entitlements
- **Notarization**: Automatic notarization via `@electron/notarize`
- **Icon**: Custom app icon at `build/icon.icns`
- **Categories**: Music category in App Store

### Backend Build

The Python backend uses PyInstaller with:

- Hidden imports for all dependencies (music2latent, transformers, librosa, etc.)
- Data file collection for model weights and configs
- Single-file executable with bundled dependencies
- Dependency preflight checks before build

## 📁 Project Structure

```
cam/
├── cam_model_v4.py              # Shared v4-xl model architecture
├── train-clap-v4-warmup-xl-cfg.py    # Training script
├── generate-clap-v4-xl-warmup-val-cfg.py  # Generation script
├── export-clap-v4-xl.py         # TorchScript export
├── preprocess-clap-v4-xl.py     # Audio preprocessing
├── electron-gui/                # Electron GUI application
│   ├── main.js                  # Electron main process
│   ├── renderer.js              # Renderer process
│   ├── backend.py               # Python generation server
│   ├── index.html               # UI HTML
│   ├── styles.css               # UI styling
│   ├── package.json             # Node dependencies
│   ├── backend.spec             # PyInstaller spec
│   └── scripts/                 # Build scripts
│       ├── build-backend.mjs    # Backend build
│       ├── prepare-build.mjs    # Build preparation
│       └── notarize.mjs         # macOS notarization
├── ts-models/                   # TorchScript models
├── checkpoints_v4/              # Training checkpoints
├── preprocessed_v4/             # Preprocessed training data
└── requirements.txt             # Python dependencies
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is provided as-is for research purposes. Please refer to the original paper for citation information.

## 📚 Citation

If you use this code in your research, please cite the original paper:

```bibtex
@inproceedings{pasini2024continuous,
  title={Continuous Autoregressive Models with Noise Augmentation Avoid Error Accumulation},
  author={Pasini, Marco and Nistal, Javier and Lattner, Stefan and Fazekas, György},
  booktitle={NeurIPS 2024 Audio Imagination Workshop},
  year={2024}
}
```

## 🙏 Acknowledgments

- Original CAM paper authors for the innovative approach
- music2latent for the audio encoder/decoder
- Hugging Face transformers for CLAP model
- PyTorch team for the excellent framework
- Electron team for the desktop app framework
