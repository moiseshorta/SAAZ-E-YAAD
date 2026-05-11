#!/usr/bin/env python

import argparse
import os
import random
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from music2latent import EncoderDecoder
from transformers import ClapModel, ClapProcessor


@torch.no_grad()
def _compute_clap_text_token(
    *,
    clap_model: ClapModel,
    clap_processor: ClapProcessor,
    device: torch.device,
    text: str,
) -> torch.Tensor:
    inputs = clap_processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    emb = clap_model.get_text_features(**inputs)
    if emb.dim() > 2:
        emb = emb.flatten(1)
    emb = emb.squeeze(0).detach().to(dtype=torch.float32)
    return emb.unsqueeze(0)


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
    max_tokens: Optional[int],
) -> torch.Tensor:
    wav, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    wav = np.asarray(wav).reshape(-1)

    window_len = int(round(float(window_seconds) * float(sample_rate)))
    hop_len = int(round(float(hop_seconds) * float(sample_rate)))
    if window_len <= 0:
        raise ValueError("window_seconds must be > 0")
    if hop_len <= 0:
        raise ValueError("hop_seconds must be > 0")

    if wav.shape[0] == 0:
        wav = np.zeros((window_len,), dtype=np.float32)

    if wav.shape[0] <= window_len:
        starts = [0]
    else:
        last = int(wav.shape[0] - window_len)
        if hop_len >= last:
            starts = [0, last]
        else:
            starts = list(range(0, last + 1, hop_len))
            if starts[-1] != last:
                starts.append(last)

    if max_tokens is not None and len(starts) > int(max_tokens):
        mt = int(max_tokens)
        if mt <= 1:
            starts = [0]
        elif mt == 2:
            starts = [starts[0], starts[-1]]
        else:
            lin = np.linspace(0, len(starts) - 1, mt)
            idxs = sorted(set(int(round(i)) for i in lin))
            starts = [starts[i] for i in idxs]
            if starts[0] != 0:
                starts[0] = 0
            if wav.shape[0] > window_len and starts[-1] != int(wav.shape[0] - window_len):
                starts[-1] = int(wav.shape[0] - window_len)

    tokens = []
    for start in starts:
        seg = wav[int(start) : int(start) + window_len]
        if seg.shape[0] < window_len:
            pad = window_len - seg.shape[0]
            seg = np.pad(seg, (0, pad), mode="constant", constant_values=0.0)
        inputs = clap_processor(audios=seg, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        emb = clap_model.get_audio_features(**inputs)
        if emb.dim() > 2:
            emb = emb.flatten(1)
        emb = emb.squeeze(0).detach().to(dtype=torch.float32)
        tokens.append(emb)

    if not tokens:
        raise RuntimeError("Failed to compute any CLAP tokens")

    return torch.stack(tokens, dim=0)


@torch.no_grad()
def generate_ts(
    *,
    ts_model: torch.jit.RecursiveScriptModule,
    device: torch.device,
    seq_len: int,
    num_diffusion_steps: int,
    seed_scale: float,
    temperature: float,
    clap_tokens: Optional[torch.Tensor],
    clap_mask: Optional[torch.Tensor],
    cfg_scale: float,
    prefix_latents: Optional[torch.Tensor],
) -> torch.Tensor:
    use_conditioning = clap_tokens is not None
    if not use_conditioning:
        clap_tokens = torch.zeros((1, 1, 512), device=device, dtype=torch.float32)
        clap_mask = torch.ones((1, 1), device=device, dtype=torch.bool)
    else:
        if clap_tokens.dim() == 2:
            clap_tokens = clap_tokens.unsqueeze(0)
        if clap_mask is None:
            clap_mask = torch.ones((clap_tokens.shape[0], clap_tokens.shape[1]), device=device, dtype=torch.bool)
        clap_tokens = clap_tokens.to(device=device, dtype=torch.float32)
        clap_mask = clap_mask.to(device=device, dtype=torch.bool)

    if prefix_latents is not None:
        prefix_latents = prefix_latents.to(device=device, dtype=torch.float32)

    return ts_model.forward(
        int(seq_len),
        clap_tokens,
        clap_mask,
        float(cfg_scale),
        int(num_diffusion_steps),
        float(seed_scale),
        float(temperature),
        prefix_latents,
        bool(use_conditioning),
    )


def _pick_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def main(args):
    device = _pick_device(args.device)

    if not str(args.ts_model).lower().endswith(".ts"):
        raise ValueError("--ts_model must point to a TorchScript file with a .ts extension")

    ts_model = torch.jit.load(args.ts_model, map_location="cpu")
    ts_model.eval()
    ts_model = ts_model.to(device)

    clap_tokens = None
    clap_mask = None

    if args.condition_type != "none":
        clap_processor = ClapProcessor.from_pretrained(args.clap_model)
        clap_model = ClapModel.from_pretrained(args.clap_model).to(device)
        clap_model.eval()

        if args.condition_type == "text":
            if args.condition_text is None:
                raise ValueError("--condition_text is required for condition_type=text")
            clap_tokens = _compute_clap_text_token(
                clap_model=clap_model,
                clap_processor=clap_processor,
                device=device,
                text=args.condition_text,
            )
            clap_mask = torch.ones((1, clap_tokens.shape[0]), dtype=torch.bool, device=device)
        elif args.condition_type == "audio":
            if args.condition_audio_path is None:
                raise ValueError("--condition_audio_path is required for condition_type=audio")
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
            clap_tokens = clap_tokens.to(device=device, dtype=torch.float32)
            clap_mask = torch.ones((1, clap_tokens.shape[0]), dtype=torch.bool, device=device)
        else:
            raise ValueError("condition_type must be one of: none, text, audio")

    samples_per_latent = int(args.samples_per_latent)
    if args.latent_fps is not None:
        samples_per_latent = int(round(float(args.sample_rate) / float(args.latent_fps)))
    latent_fps = float(args.sample_rate) / float(samples_per_latent)

    prompt_latents = None
    prompt_frames = 0
    if args.prompt_audio_path is not None:
        wav, _ = librosa.load(args.prompt_audio_path, sr=args.sample_rate, mono=False)
        max_waveform_length = None
        if args.prompt_seconds is not None:
            max_waveform_length = int(float(args.prompt_seconds) * float(args.sample_rate))
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
            raise ValueError("music2latent.encode returned unexpected shape")
        prompt_frames = int(prompt_latents.shape[2])

    os.makedirs(args.out_dir, exist_ok=True)

    for i in range(int(args.num_samples)):
        if args.seed is not None:
            sample_seed = int(args.seed) + i
        else:
            sample_seed = random.randint(0, 2**32 - 1)

        torch.manual_seed(sample_seed)
        np.random.seed(sample_seed)
        random.seed(sample_seed)

        total_seq_len = int(args.gen_seq_len)
        if prompt_latents is not None and args.continue_seconds is not None:
            continue_frames = int(round(float(args.continue_seconds) * latent_fps))
            total_seq_len = int(prompt_frames + max(0, continue_frames))
        if prompt_latents is not None and total_seq_len < prompt_frames:
            total_seq_len = int(prompt_frames)

        generated_latents = generate_ts(
            ts_model=ts_model,
            device=device,
            seq_len=total_seq_len,
            num_diffusion_steps=int(args.num_diffusion_steps),
            seed_scale=float(args.seed_scale),
            temperature=float(args.temperature),
            clap_tokens=clap_tokens,
            clap_mask=clap_mask,
            cfg_scale=float(args.cfg_scale),
            prefix_latents=prompt_latents,
        )

        if isinstance(generated_latents, torch.Tensor):
            generated_latents = generated_latents.detach().to("cpu")

        encoder_decoder = EncoderDecoder()
        max_waveform_length = int(total_seq_len * samples_per_latent)
        decoded_audio = encoder_decoder.decode(generated_latents, max_waveform_length=max_waveform_length)
        if isinstance(decoded_audio, torch.Tensor):
            decoded_audio = decoded_audio.cpu().numpy()
        if decoded_audio.shape[0] < decoded_audio.shape[1]:
            decoded_audio = decoded_audio.T

        out_path = os.path.join(args.out_dir, f"generated_ts_{i}_seed_{sample_seed}.wav")
        sf.write(out_path, decoded_audio, int(args.sample_rate))
        print(f"Saved generated audio to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ts_model", type=str, required=True)

    parser.add_argument("--sample_rate", type=int, default=48_000)
    parser.add_argument("--samples_per_latent", type=int, default=4096)
    parser.add_argument("--latent_fps", type=float, default=None)

    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--channels", type=int, default=2)

    parser.add_argument("--gen_seq_len", type=int, default=128)
    parser.add_argument("--num_diffusion_steps", type=int, default=16)
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

    parser.add_argument("--prompt_audio_path", type=str, default=None)
    parser.add_argument("--prompt_seconds", type=float, default=None)
    parser.add_argument("--continue_seconds", type=float, default=None)

    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="./inference_samples_ts")
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    main(args)
