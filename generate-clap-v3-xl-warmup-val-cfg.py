#!/usr/bin/env python

"""Generate audio with the v3-xl CLAP-conditioned CAM model.

This is an inference-only script. It:
- Loads a trained checkpoint (optionally using EMA weights).
- Builds a conditioning signal from either:
  - an audio clip (multi-token CLAP embeddings), or
  - a text prompt (single-token CLAP embedding), or
  - no conditioning.
- Generates a sequence of `music2latent` frames autoregressively.
- Decodes the latent sequence back to waveform audio.

The model uses a Transformer backbone plus a diffusion-like inner sampler loop per frame.
Conditioning is provided as a *sequence* of CLAP tokens with a boolean mask.

This script also supports "audio autocomplete": you can provide a prompt audio file to
seed the first N latent frames, and the model will continue generating beyond that prefix.
"""

import argparse
import os
import random

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from transformers import ClapModel, ClapProcessor

from music2latent import EncoderDecoder


def rotate_half(x):
    """Helper for RoPE: swap and negate half of the last dimension."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, freqs):
    """Apply Rotary Positional Embeddings (RoPE) to a query/key tensor."""
    return (x * freqs.cos()) + (rotate_half(x) * freqs.sin())


def get_rotary_embedding(seq_len, dim):
    """Create the RoPE frequency matrix for a given sequence length and head dimension."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len, dtype=torch.float)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    freqs = torch.cat([freqs, freqs], dim=-1)
    return freqs.unsqueeze(1)


class AdaLayerNorm(nn.Module):
    """LayerNorm with scale/bias conditioned on a scalar noise level `sigma`."""

    def __init__(self, normalized_shape):
        super(AdaLayerNorm, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape)
        self.scale_proj = nn.Linear(1, normalized_shape)
        self.bias_proj = nn.Linear(1, normalized_shape)

    def forward(self, x, sigma):
        gamma = 1 + self.scale_proj(sigma)
        beta = self.bias_proj(sigma)
        return self.ln(x) * gamma + beta


class RotaryMultiheadAttention(nn.Module):
    """Multi-head self-attention with Rotary Positional Embeddings (RoPE)."""

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(RotaryMultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

    def forward(self, query, key, value, attn_mask=None):
        seq_len, batch, _ = query.size()
        q = (
            query.view(seq_len, batch, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(seq_len, batch * self.num_heads, self.head_dim)
        )
        k = (
            key.view(seq_len, batch, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(seq_len, batch * self.num_heads, self.head_dim)
        )
        freqs = get_rotary_embedding(seq_len, self.head_dim).to(query.device)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)
        q = (
            q.view(seq_len, batch, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(seq_len, batch, self.embed_dim)
        )
        k = (
            k.view(seq_len, batch, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(seq_len, batch, self.embed_dim)
        )
        out, weights = self.attn(q, k, value, attn_mask=attn_mask)
        return out, weights


class TransformerBlock(nn.Module):
    """A standard Transformer block: self-attention + feed-forward network."""

    def __init__(self, model_dim, num_heads, ff_mult=4, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(model_dim)
        self.attn = RotaryMultiheadAttention(model_dim, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, model_dim * ff_mult),
            nn.GELU(),
            nn.Linear(model_dim * ff_mult, model_dim),
        )

    def forward(self, x, attn_mask):
        residual = x
        x = self.ln1(x)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = residual + attn_out
        residual = x
        x = self.ln2(x)
        x = residual + self.ff(x)
        return x


class CrossAttention(nn.Module):
    """Cross-attention from latent sequence (queries) to conditioning tokens (keys/values)."""

    def __init__(self, query_dim, context_dim, num_heads, dropout=0.0):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, context_mask=None):
        """Attend from `x` to `context`.

        Inputs:
        - `x`: latent/hidden sequence (B, T, query_dim)
        - `context`: conditioning tokens (B, L, context_dim)
        - `context_mask`: which conditioning tokens are valid (B, L)
        """
        B, T, _ = x.shape
        L = context.shape[1]
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if context_mask is not None:
            m = context_mask.to(dtype=torch.bool, device=attn_scores.device)
            m = m.view(B, 1, 1, L)
            attn_scores = attn_scores.masked_fill(~m, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(attn_out)


class Backbone(nn.Module):
    """Autoregressive Transformer backbone operating on latent frames."""

    def __init__(self, latent_dim=128, model_dim=768, num_layers=16, num_heads=4, max_seq_len=512):
        super(Backbone, self).__init__()
        self.input_proj = nn.Linear(latent_dim, model_dim)
        self.layers = nn.ModuleList([TransformerBlock(model_dim, num_heads, ff_mult=4) for _ in range(num_layers)])
        self.model_dim = model_dim

    def forward(self, x):
        """Run a causal Transformer over the latent sequence."""
        B, T, _ = x.shape
        x = self.input_proj(x)
        x = x.transpose(0, 1)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        for layer in self.layers:
            x = layer(x, attn_mask=mask)
        x = x.transpose(0, 1)
        return x


class SamplerMLP(nn.Module):
    """MLP used inside the diffusion-like sampler (inference-time denoising loop)."""

    def __init__(self, cond_dim, latent_dim=128, hidden_dim=768, num_layers=8, mlp_mult=4, dropout=0.0):
        super(SamplerMLP, self).__init__()
        self.input_dim = cond_dim + latent_dim
        self.hidden_dim = hidden_dim
        layers = []
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        layers.append(AdaLayerNorm(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(AdaLayerNorm(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, sigma):
        """Predict denoising updates given features `x` and noise level `sigma`."""
        B, T, _ = x.shape
        x = x.view(B * T, -1)
        sigma = sigma.view(B * T, 1)
        for layer in self.layers:
            if isinstance(layer, AdaLayerNorm):
                x = layer(x, sigma)
            else:
                x = layer(x)
        x = x.view(B, T, -1)
        return x


def masked_mean(x, mask, dim=1, eps=1e-8):
    """Mean-pool a sequence with a boolean mask."""
    m = mask.to(dtype=x.dtype)
    num = (x * m.unsqueeze(-1)).sum(dim=dim)
    den = m.sum(dim=dim).clamp_min(eps).unsqueeze(-1)
    return num / den


class CAMModel(nn.Module):
    """Inference-time model container.

    The training script defines the same architectural components. For generation we only
    need:
    - `backbone` (autoregressive Transformer)
    - `cross_attn` (inject CLAP context)
    - `sampler` (diffusion-like inner loop)
    along with learned special tokens (`null_clap`, `zSOS`).
    """

    def __init__(self, latent_dim=64, model_dim=768, seq_len=128, channels=2, clap_dim=512, cfg_drop_prob=0.0):
        super(CAMModel, self).__init__()
        self.original_latent_dim = latent_dim
        self.channels = channels
        self.latent_dim = latent_dim * channels
        self.model_dim = model_dim
        self.seq_len = seq_len
        self.cfg_drop_prob = cfg_drop_prob

        self.null_clap = nn.Parameter(torch.empty(clap_dim))
        nn.init.xavier_uniform_(self.null_clap.unsqueeze(0))

        self.backbone = Backbone(latent_dim=self.latent_dim, model_dim=model_dim, num_layers=24, num_heads=6, max_seq_len=seq_len)
        self.clap_proj = nn.Linear(clap_dim, model_dim)
        self.cross_attn = CrossAttention(query_dim=model_dim, context_dim=model_dim, num_heads=4)
        self.sampler = SamplerMLP(cond_dim=2 * model_dim, latent_dim=self.latent_dim, hidden_dim=model_dim, num_layers=12)
        self.zSOS = nn.Parameter(torch.randn(1, model_dim).squeeze(0))


@torch.no_grad()
def _compute_clap_audio_tokens(
    *,
    clap_model: ClapModel,
    clap_processor: ClapProcessor,
    device: torch.device,
    audio_path: str,
    sample_rate: int,
    window_seconds: float,
    hop_seconds: float,
    max_tokens: int | None,
) -> torch.Tensor:
    """Compute multi-token CLAP conditioning from an audio file.

    Returns a tensor of shape (L, 512), where each row is a CLAP embedding computed from
    a time window of the audio. This allows longer audio prompts to be represented with
    multiple tokens.
    """
    wav, _ = librosa.load(audio_path, sr=sample_rate, mono=True)

    window_len = int(round(float(window_seconds) * float(sample_rate)))
    hop_len = int(round(float(hop_seconds) * float(sample_rate)))
    if window_len <= 0:
        raise ValueError("clap_window_seconds must be > 0")
    if hop_len <= 0:
        raise ValueError("clap_hop_seconds must be > 0")

    if wav.ndim != 1:
        wav = np.asarray(wav).reshape(-1)

    tokens: list[torch.Tensor] = []
    if wav.shape[0] == 0:
        wav = np.zeros((window_len,), dtype=np.float32)

    for start in range(0, wav.shape[0], hop_len):
        if max_tokens is not None and len(tokens) >= int(max_tokens):
            break

        seg = wav[start : start + window_len]
        if seg.shape[0] < window_len:
            pad = window_len - seg.shape[0]
            seg = np.pad(seg, (0, pad), mode="constant", constant_values=0.0)

        inputs = clap_processor(audios=seg, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        emb = clap_model.get_audio_features(**inputs)
        if emb.dim() > 2:
            emb = emb.flatten(1)
        emb = emb.squeeze(0).detach().to(dtype=torch.float32).cpu()
        tokens.append(emb)

        if start + window_len >= wav.shape[0]:
            break

    if not tokens:
        raise ValueError(f"Failed to compute any CLAP tokens for {os.path.basename(audio_path)}")

    return torch.stack(tokens, dim=0)


@torch.no_grad()
def _compute_clap_text_token(
    *,
    clap_model: ClapModel,
    clap_processor: ClapProcessor,
    device: torch.device,
    text: str,
) -> torch.Tensor:
    """Compute a single CLAP token from a text prompt.

    Returns a tensor of shape (1, 512) to match the multi-token audio path.
    """
    inputs = clap_processor(text=[text], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    emb = clap_model.get_text_features(**inputs)
    if emb.dim() > 2:
        emb = emb.flatten(1)
    emb = emb.squeeze(0).detach().to(dtype=torch.float32).cpu()
    return emb.unsqueeze(0)


def generate_sample(
    model,
    device,
    seq_len,
    num_diffusion_steps=48,
    seed_scale=0.0,
    temperature=1.0,
    clap_tokens=None,
    clap_mask=None,
    cfg_scale=1.0,
    prefix_latents=None,
):
    """Generate a latent sequence with optional CLAP conditioning and audio prefix.

    - `clap_tokens` / `clap_mask` provide multi-token conditioning (B=1).
    - `cfg_scale` applies classifier-free guidance by mixing conditional and unconditional
      sampler predictions.
    - `prefix_latents` enables "audio autocomplete": pre-fill the first frames with
      encoded prompt latents, then continue generating.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            generated = []

            if clap_tokens is None:
                clap_tokens = model.null_clap.view(1, 1, -1)
                clap_mask = torch.ones((1, 1), dtype=torch.bool)
            else:
                if isinstance(clap_tokens, torch.Tensor) and clap_tokens.dim() == 2:
                    clap_tokens = clap_tokens.unsqueeze(0)
                if clap_mask is None:
                    clap_mask = torch.ones((clap_tokens.shape[0], clap_tokens.shape[1]), dtype=torch.bool)
                elif isinstance(clap_mask, torch.Tensor) and clap_mask.dim() == 1:
                    clap_mask = clap_mask.unsqueeze(0)

            clap_tokens = clap_tokens.to(device=device, dtype=torch.float32)
            clap_mask = clap_mask.to(device=device, dtype=torch.bool)

            cond_ctx = model.clap_proj(clap_tokens)
            cond_pool = masked_mean(cond_ctx, clap_mask, dim=1).unsqueeze(1)

            uncond_tokens = model.null_clap.view(1, 1, -1).to(device=device, dtype=torch.float32)
            uncond_mask = torch.ones((1, 1), device=device, dtype=torch.bool)
            uncond_ctx = model.clap_proj(uncond_tokens)
            uncond_pool = masked_mean(uncond_ctx, uncond_mask, dim=1).unsqueeze(1)

            if prefix_latents is not None:
                # Accept prefix in either music2latent format (C, D, T) or flattened (T, latent_dim).
                if isinstance(prefix_latents, torch.Tensor) and prefix_latents.dim() == 3:
                    if prefix_latents.shape[0] != model.channels:
                        if prefix_latents.shape[0] == 1 and model.channels == 2:
                            prefix_latents = prefix_latents.repeat(2, 1, 1)
                        else:
                            raise ValueError(
                                f"prefix_latents has {prefix_latents.shape[0]} channels but model expects {model.channels}."
                            )
                    prefix_latents = prefix_latents.to(device=device, dtype=torch.float32)
                    prefix_frames = prefix_latents.permute(2, 0, 1).reshape(prefix_latents.shape[2], -1)
                elif isinstance(prefix_latents, torch.Tensor) and prefix_latents.dim() == 2:
                    prefix_frames = prefix_latents.to(device=device, dtype=torch.float32)
                else:
                    raise ValueError("prefix_latents must be a Tensor with shape (channels, 64, T) or (T, latent_dim).")

                if prefix_frames.shape[1] != model.latent_dim:
                    raise ValueError(
                        f"prefix_latents has latent_dim={prefix_frames.shape[1]} but model expects {model.latent_dim}."
                    )

                prefix_len = min(int(prefix_frames.shape[0]), int(seq_len))
                for i in range(prefix_len):
                    generated.append(prefix_frames[i : i + 1, :])

            with Progress(
                TextColumn("[bold magenta]{task.description}"),
                BarColumn(bar_width=40, style="magenta"),
                TextColumn("{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("Generating Latents", total=seq_len)
                if len(generated) > 0:
                    progress.update(task, advance=len(generated))

                if len(generated) == 0:
                    # First frame: sample from scratch using the sampler inner loop.
                    z0 = model.zSOS.unsqueeze(0).unsqueeze(0)
                    if seed_scale:
                        z0 = z0 + seed_scale * torch.randn_like(z0)

                    x = torch.randn(1, 1, model.latent_dim, device=device)
                    noise_level = torch.ones(1, 1, 1, device=device)
                    for _ in range(num_diffusion_steps):
                        step_size = 1.0 / num_diffusion_steps
                        inp_cond = torch.cat([z0, x, cond_pool], dim=-1)
                        eps_cond = model.sampler(inp_cond, noise_level)
                        inp_uncond = torch.cat([z0, x, uncond_pool], dim=-1)
                        eps_uncond = model.sampler(inp_uncond, noise_level)
                        eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
                        x = x + step_size * (eps / temperature)
                        noise_level = noise_level - step_size
                    generated.append(x.squeeze(0))
                    progress.update(task, advance=1)

                start_t = len(generated)
                for _t in range(start_t, seq_len):
                    # Autoregressive step: encode previous frames, then sample the next one.
                    prev_seq = torch.stack(generated, dim=1)
                    backbone_out = model.backbone(prev_seq)
                    backbone_out_cond = backbone_out + model.cross_attn(backbone_out, cond_ctx, context_mask=clap_mask)
                    backbone_out_uncond = backbone_out + model.cross_attn(backbone_out, uncond_ctx, context_mask=uncond_mask)
                    z_t_cond = backbone_out_cond[:, -1:, :]
                    z_t_uncond = backbone_out_uncond[:, -1:, :]

                    x = torch.randn(1, 1, model.latent_dim, device=device)
                    noise_level = torch.ones(1, 1, 1, device=device)
                    for _ in range(num_diffusion_steps):
                        step_size = 1.0 / num_diffusion_steps
                        inp_cond = torch.cat([z_t_cond, x, cond_pool], dim=-1)
                        eps_cond = model.sampler(inp_cond, noise_level)
                        inp_uncond = torch.cat([z_t_uncond, x, uncond_pool], dim=-1)
                        eps_uncond = model.sampler(inp_uncond, noise_level)
                        eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
                        x = x + step_size * (eps / temperature)
                        noise_level = noise_level - step_size

                    generated.append(x.squeeze(0))
                    progress.update(task, advance=1)

            gen_seq = torch.stack(generated, dim=0)
            gen_seq = gen_seq.view(seq_len, model.channels, model.original_latent_dim)
            generated_latents = gen_seq.permute(1, 2, 0)
            return generated_latents
    finally:
        if was_training:
            model.train()


def main(args):
    """CLI entrypoint: load model/conditioning and write generated waveforms to disk."""
    if getattr(args, "device", None) is None or args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    samples_per_latent = int(getattr(args, "samples_per_latent", 4096))
    if getattr(args, "latent_fps", None) is not None:
        samples_per_latent = int(round(float(args.sample_rate) / float(args.latent_fps)))
    if samples_per_latent <= 0:
        raise ValueError("samples_per_latent must be > 0")
    latent_fps = float(args.sample_rate) / float(samples_per_latent)

    print(f"Using device: {device}")

    print(f"Loading CAMModel checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if getattr(args, "use_ema", False):
        # If available, EMA weights often give smoother/better generations.
        ema_sd = checkpoint.get("ema_state_dict", None) if isinstance(checkpoint, dict) else None
        if ema_sd is not None:
            print("Using EMA weights from checkpoint (ema_state_dict).")
            state_dict = ema_sd
        else:
            print("[WARN] --use_ema was set but checkpoint has no ema_state_dict. Using model_state_dict instead.")

    model = CAMModel(latent_dim=64, model_dim=768, seq_len=args.segment_frames, channels=2, clap_dim=512, cfg_drop_prob=0.0)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)
    model.to(device)
    model.eval()

    clap_tokens = None
    clap_mask = None

    if args.condition_type != "none":
        # Conditioning is computed at inference time (not loaded from dataset).
        clap_processor = ClapProcessor.from_pretrained(args.clap_model)
        clap_model = ClapModel.from_pretrained(args.clap_model).to(device)
        clap_model.eval()

        if args.condition_type == "audio":
            if args.condition_audio_path is None:
                raise ValueError("For condition_type 'audio', provide --condition_audio_path.")
            # Multi-token CLAP conditioning from an audio clip.
            clap_tokens = _compute_clap_audio_tokens(
                clap_model=clap_model,
                clap_processor=clap_processor,
                device=device,
                audio_path=args.condition_audio_path,
                sample_rate=args.sample_rate,
                window_seconds=args.clap_window_seconds,
                hop_seconds=args.clap_hop_seconds,
                max_tokens=args.clap_max_tokens,
            )
            clap_mask = torch.ones((1, clap_tokens.shape[0]), dtype=torch.bool)
        elif args.condition_type == "text":
            if args.condition_text is None:
                raise ValueError("For condition_type 'text', provide --condition_text.")
            # Single-token CLAP conditioning from text.
            clap_tokens = _compute_clap_text_token(
                clap_model=clap_model,
                clap_processor=clap_processor,
                device=device,
                text=args.condition_text,
            )
            clap_mask = torch.ones((1, clap_tokens.shape[0]), dtype=torch.bool)
        else:
            raise ValueError("condition_type must be one of: audio, text, none")

    if clap_tokens is not None:
        if clap_tokens.dim() != 2 or clap_tokens.shape[1] != 512:
            raise ValueError(f"Expected clap_tokens shape (N, 512), got {tuple(clap_tokens.shape)}")
        clap_tokens = clap_tokens.to(dtype=torch.float32)
        clap_mask = clap_mask.to(dtype=torch.bool)

    if getattr(args, "continue_seconds", None) is not None and not getattr(args, "prompt_audio_path", None):
        raise ValueError("--continue_seconds requires --prompt_audio_path")

    prompt_latents = None
    prompt_frames = 0
    if getattr(args, "prompt_audio_path", None):
        # Audio autocomplete: encode a prompt audio prefix into latents.
        print(f"Encoding prompt audio with music2latent from: {args.prompt_audio_path}")
        wav, _ = librosa.load(args.prompt_audio_path, sr=args.sample_rate, mono=False)
        max_waveform_length = None
        if getattr(args, "prompt_seconds", None) is not None:
            max_waveform_length = int(float(args.prompt_seconds) * args.sample_rate)
            if isinstance(wav, np.ndarray):
                if wav.ndim == 1:
                    wav = wav[:max_waveform_length]
                elif wav.ndim == 2:
                    wav = wav[:, :max_waveform_length]
        encoder_decoder = EncoderDecoder()
        with torch.no_grad():
            prompt_latents = encoder_decoder.encode(wav, max_waveform_length=None)
        if isinstance(prompt_latents, torch.Tensor):
            prompt_latents = prompt_latents.detach().to(dtype=torch.float32)
        if prompt_latents.dim() != 3:
            raise ValueError(f"music2latent.encode returned unexpected shape: {tuple(prompt_latents.shape)}")
        prompt_frames = int(prompt_latents.shape[2])
        print(f"Prompt latents shape: {tuple(prompt_latents.shape)}")

    for i in range(args.num_samples):
        if args.seed is not None:
            sample_seed = args.seed + i
        else:
            sample_seed = random.randint(0, 2**32 - 1)

        torch.manual_seed(sample_seed)
        np.random.seed(sample_seed)
        random.seed(sample_seed)
        print(f"Using seed: {sample_seed} for sample {i}")

        total_seq_len = args.gen_seq_len
        if prompt_latents is not None and getattr(args, "continue_seconds", None) is not None:
            # If continue_seconds is provided, total length is prompt + continuation.
            continue_frames = int(round(float(args.continue_seconds) * latent_fps))
            total_seq_len = int(prompt_frames + max(0, continue_frames))
        if prompt_latents is not None and total_seq_len < prompt_frames:
            total_seq_len = int(prompt_frames)

        generated_latents = generate_sample(
            model,
            device,
            seq_len=total_seq_len,
            num_diffusion_steps=args.num_diffusion_steps,
            seed_scale=args.seed_scale,
            temperature=args.temperature,
            clap_tokens=clap_tokens,
            clap_mask=clap_mask,
            cfg_scale=args.cfg_scale,
            prefix_latents=prompt_latents,
        )

        if isinstance(generated_latents, torch.Tensor):
            generated_latents = generated_latents.detach().to("cpu")

        print(f"Decoding latent sequence into audio for sample {i}...")
        encoder_decoder = EncoderDecoder()
        max_waveform_length = int(total_seq_len * samples_per_latent)
        decoded_audio = encoder_decoder.decode(generated_latents, max_waveform_length=max_waveform_length)
        if isinstance(decoded_audio, torch.Tensor):
            decoded_audio = decoded_audio.cpu().numpy()
        if decoded_audio.shape[0] < decoded_audio.shape[1]:
            decoded_audio = decoded_audio.T

        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, f"generated_sample_{i}_seed_{sample_seed}.wav")
        sf.write(out_path, decoded_audio, args.sample_rate)
        print(f"Saved generated audio to: {out_path}")


if __name__ == "__main__":
    # CLI entrypoint.
    parser = argparse.ArgumentParser(description="Inference v3-xl: CAM with multi-token CLAP conditioning.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--use_ema", action="store_true")

    parser.add_argument("--sample_rate", type=int, default=48_000)
    parser.add_argument("--segment_frames", type=int, default=128)

    parser.add_argument("--gen_seq_len", type=int, default=256)
    parser.add_argument("--prompt_audio_path", type=str, default=None)
    parser.add_argument("--prompt_seconds", type=float, default=None)
    parser.add_argument("--continue_seconds", type=float, default=None)

    parser.add_argument("--samples_per_latent", type=int, default=4096)
    parser.add_argument("--latent_fps", type=float, default=None)

    parser.add_argument("--num_diffusion_steps", type=int, default=48)
    parser.add_argument("--seed_scale", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg_scale", type=float, default=3.0)

    parser.add_argument("--condition_type", type=str, choices=["audio", "text", "none"], default="none")
    parser.add_argument("--condition_audio_path", type=str, default=None)
    parser.add_argument("--condition_text", type=str, default=None)

    parser.add_argument("--clap_model", type=str, default="laion/larger_clap_music")
    parser.add_argument("--clap_window_seconds", type=float, default=10.0)
    parser.add_argument("--clap_hop_seconds", type=float, default=10.0)
    parser.add_argument("--clap_max_tokens", type=int, default=None)

    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="./inference_samples")
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    main(args)
