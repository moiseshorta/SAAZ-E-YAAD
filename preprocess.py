#!/usr/bin/env python

"""Preprocess audio for v3 CLAP-conditioned CAM training.

This script converts raw audio files into a compact training format consumed by
`train-clap-v3-warmup-xl-cfg.py`.

It produces one `.pt` file per audio track containing:
- `music2latent`: the encoded latent sequence from `music2latent` (shape roughly (C, D, T))
- `clap_tokens`: CLAP conditioning tokens aligned to latent *segments*
  (shape (num_segments, L, 512))
- `clap_mask`: boolean mask for valid CLAP tokens (shape (num_segments, L))

Why segment alignment matters:
During training we sample fixed-length latent chunks of length `segment_frames`. If CLAP
conditioning is computed over a different time span (e.g., whole-file), the model sees a
mismatch between what it is asked to predict and what it is conditioned on. This script
ensures each latent segment uses CLAP tokens computed from the corresponding audio slice.
"""

import argparse
import glob
import os
from typing import Optional

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor

from music2latent import EncoderDecoder


def _list_audio_files(audio_dir: str) -> list[str]:
    """List supported audio files in a directory."""
    wav_files = glob.glob(os.path.join(audio_dir, "*.[Ww][Aa][Vv]"))
    mp3_files = glob.glob(os.path.join(audio_dir, "*.[Mm][Pp]3"))
    flac_files = glob.glob(os.path.join(audio_dir, "*.[Ff][Ll][Aa][Cc]"))
    return sorted(set(wav_files + mp3_files + flac_files))


def _ensure_channels_first(wav: np.ndarray) -> np.ndarray:
    if not isinstance(wav, np.ndarray):
        wav = np.asarray(wav)
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32, copy=False)
    if wav.ndim == 2:
        c0, c1 = int(wav.shape[0]), int(wav.shape[1])
        if c0 > 16 and c1 <= 16:
            wav = wav.T
        elif c0 > c1 and c1 <= 64:
            wav = wav.T
    return wav


@torch.no_grad()
def _compute_clap_audio_tokens(
    *,
    clap_model: ClapModel,
    clap_processor: ClapProcessor,
    device: torch.device,
    sample_rate: int,
    wav: np.ndarray,
    window_seconds: float,
    hop_seconds: float,
    max_tokens: Optional[int],
) -> torch.Tensor:
    """Compute a sequence of CLAP audio embeddings for a waveform.

    We slide a fixed-length window (`window_seconds`) across the waveform with hop
    (`hop_seconds`) and run CLAP on each window.

    Returns:
    - Tensor of shape (L, 512), where L is the number of windows/tokens.
    """
    window_len = int(round(float(window_seconds) * float(sample_rate)))
    hop_len = int(round(float(hop_seconds) * float(sample_rate)))
    if window_len <= 0:
        raise ValueError("window_seconds must be > 0")
    if hop_len <= 0:
        raise ValueError("hop_seconds must be > 0")

    if wav.ndim != 1:
        wav = np.asarray(wav).reshape(-1)

    tokens: list[torch.Tensor] = []
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

    for start in starts:
        # Extract a fixed-length window (pad with zeros if we fall off the end).
        seg = wav[int(start) : int(start) + window_len]
        if seg.shape[0] < window_len:
            pad = window_len - seg.shape[0]
            seg = np.pad(seg, (0, pad), mode="constant", constant_values=0.0)

        # CLAP processor builds model inputs (mel/features + normalization).
        inputs = clap_processor(audios=seg, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        emb = clap_model.get_audio_features(**inputs)
        if emb.dim() > 2:
            emb = emb.flatten(1)
        emb = emb.squeeze(0).detach().to(dtype=torch.float32).cpu()  # (512,)
        tokens.append(emb)

    if not tokens:
        raise ValueError("Failed to compute any CLAP tokens")

    out = torch.stack(tokens, dim=0)  # (N, 512)
    return out


@torch.no_grad()
def _compute_music2latent(
    *,
    encoder: EncoderDecoder,
    sample_rate: int,
    wav: np.ndarray,
    max_seconds: Optional[float],
) -> torch.Tensor:
    """Compute music2latent latents from a waveform.

    Notes:
    - `music2latent` internally expects 48kHz audio.
    - If `max_seconds` is set, we truncate the waveform before encoding.
    """
    max_waveform_length: Optional[int]
    if max_seconds is None:
        max_waveform_length = None
    else:
        max_waveform_length = int(max_seconds * sample_rate)

    if max_waveform_length is not None:
        if isinstance(wav, np.ndarray):
            if wav.ndim == 1:
                wav = wav[:max_waveform_length]
            elif wav.ndim == 2:
                wav = wav[:, :max_waveform_length]

    wav = _ensure_channels_first(wav)

    latents = encoder.encode(wav, max_waveform_length=max_waveform_length)

    return latents.detach().to(dtype=torch.float32).cpu()


def main() -> None:
    """CLI entrypoint for preprocessing."""
    p = argparse.ArgumentParser(
        description="Preprocess audio into {music2latent, clap_tokens} .pt files for train-clap-v3-warmup-xl-cfg.py"
    )
    p.add_argument("--audio_dir", required=True, type=str)
    p.add_argument("--preprocessed_dir", required=True, type=str)
    p.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="If set, only preprocess the first N files (useful for smoke tests / debugging).",
    )
    p.add_argument("--sample_rate", type=int, default=48_000)
    p.add_argument("--samples_per_latent", type=int, default=4096)
    p.add_argument("--segment_frames", type=int, default=128)
    p.add_argument(
        "--max_seconds",
        type=float,
        default=None,
        help="If set, truncate audio to this many seconds for music2latent encoding only.",
    )
    p.add_argument(
        "--encode_chunk_seconds",
        type=float,
        default=120.0,
        help="If > 0, preprocess audio in chunks of this duration (seconds) to avoid timeouts on very long files.",
    )
    p.add_argument(
        "--load_max_seconds",
        type=float,
        default=None,
        help="If set, only decode up to this many seconds from each audio file.",
    )
    p.add_argument(
        "--load_offset_seconds",
        type=float,
        default=0.0,
        help="Start decoding each audio file from this offset (in seconds).",
    )
    p.add_argument(
        "--clap_window_seconds",
        type=float,
        default=10.0,
        help="Window length in seconds for each CLAP token.",
    )
    p.add_argument(
        "--clap_hop_seconds",
        type=float,
        default=None,
        help="Hop length in seconds between CLAP windows within each segment. If unset, uses a hop that covers the segment start/end.",
    )
    p.add_argument(
        "--clap_max_tokens",
        type=int,
        default=None,
        help="If set, cap the number of CLAP tokens per file.",
    )
    p.add_argument(
        "--clap_batch_size",
        type=int,
        default=4,
        help="Batch size for CLAP window inference (larger is faster but uses more memory).",
    )
    p.add_argument(
        "--clap_model",
        type=str,
        default="laion/larger_clap_music",
        help="HuggingFace model id.",
    )
    p.add_argument("--device", type=str, default=None, help="cpu | mps | cuda | auto (default).")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    os.makedirs(args.preprocessed_dir, exist_ok=True)

    audio_files = _list_audio_files(args.audio_dir)
    if not audio_files:
        raise ValueError(f"No audio files found in {os.path.abspath(args.audio_dir)}")

    if args.max_files is not None:
        mf = int(args.max_files)
        if mf <= 0:
            raise ValueError("--max_files must be > 0")
        audio_files = audio_files[:mf]

    if args.device is None or args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    clap_processor = ClapProcessor.from_pretrained(args.clap_model)
    clap_model = ClapModel.from_pretrained(args.clap_model).to(device)
    clap_model.eval()

    encoder = EncoderDecoder()

    for fpath in tqdm(audio_files, desc="Preprocessing (music2latent + CLAP tokens)"):
        base = os.path.splitext(os.path.basename(fpath))[0]
        out_path = os.path.join(args.preprocessed_dir, base + ".pt")

        if os.path.exists(out_path) and not args.overwrite:
            continue

        load_duration = args.load_max_seconds
        if load_duration is None:
            load_duration = args.max_seconds
        load_offset = float(args.load_offset_seconds) if args.load_offset_seconds is not None else 0.0

        segment_frames = int(args.segment_frames)
        if segment_frames <= 0:
            raise ValueError("segment_frames must be > 0")

        samples_per_latent = int(args.samples_per_latent)
        if samples_per_latent <= 0:
            raise ValueError("samples_per_latent must be > 0")

        segment_samples = int(segment_frames * samples_per_latent)
        music2latent_pad_samples = 1536
        window_len = int(round(float(args.clap_window_seconds) * float(args.sample_rate)))
        if window_len <= 0:
            raise ValueError("clap_window_seconds must be > 0")

        if args.clap_hop_seconds is None:
            hop_len = max(1, segment_samples - window_len) if segment_samples > window_len else window_len
        else:
            hop_len = int(round(float(args.clap_hop_seconds) * float(args.sample_rate)))
            if hop_len <= 0:
                raise ValueError("clap_hop_seconds must be > 0")

        def _starts_for_fixed_len(wav_len: int, window_len: int, hop_len: int, max_tokens: Optional[int]) -> list[int]:
            if wav_len <= window_len:
                starts = [0]
            else:
                last = int(wav_len - window_len)
                if hop_len >= last:
                    starts = [0, last]
                else:
                    starts = list(range(0, last + 1, hop_len))
                    if starts[-1] != last:
                        starts.append(last)

            starts = sorted(set(int(s) for s in starts))

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
                    if wav_len > window_len and starts[-1] != int(wav_len - window_len):
                        starts[-1] = int(wav_len - window_len)
            return starts

        starts_template = _starts_for_fixed_len(
            int(segment_samples),
            int(window_len),
            int(hop_len),
            args.clap_max_tokens,
        )
        if len(starts_template) <= 0:
            raise ValueError("Failed to compute CLAP window starts")

        encode_chunk_seconds = float(args.encode_chunk_seconds) if args.encode_chunk_seconds is not None else 0.0
        clap_batch_size = int(max(1, args.clap_batch_size))
        tokens_per_segment = int(len(starts_template))

        latents_chunks: list[torch.Tensor] = []
        clap_chunks: list[torch.Tensor] = []

        segment_seconds = float(segment_samples) / float(args.sample_rate)

        chunk_offset = float(max(0.0, load_offset))
        total_end = None
        if load_duration is not None:
            total_end = float(max(0.0, load_offset) + float(load_duration))

        while True:
            if total_end is not None and chunk_offset >= float(total_end):
                break

            chunk_duration = None
            if encode_chunk_seconds > 0.0:
                target_content_samples = int(round(float(encode_chunk_seconds) * float(args.sample_rate)))
                aligned_content_samples = int(
                    max(segment_samples, (target_content_samples // segment_samples) * segment_samples)
                )
                if total_end is not None:
                    remaining = float(total_end) - float(chunk_offset)
                    if remaining <= 0.0:
                        break
                    remaining_samples = int(round(float(remaining) * float(args.sample_rate)))
                    remaining_content_samples = int(remaining_samples - int(music2latent_pad_samples))
                    aligned_content_samples = int(min(aligned_content_samples, remaining_content_samples))
                    aligned_content_samples = int((aligned_content_samples // segment_samples) * segment_samples)
                if aligned_content_samples < segment_samples:
                    break
                chunk_duration = float(aligned_content_samples + int(music2latent_pad_samples)) / float(args.sample_rate)
            else:
                if total_end is not None:
                    chunk_duration = float(max(0.0, float(total_end) - float(chunk_offset)))

            wav, _ = librosa.load(
                fpath,
                sr=args.sample_rate,
                mono=False,
                offset=float(chunk_offset),
                duration=chunk_duration,
            )
            if not isinstance(wav, np.ndarray) or wav.size == 0:
                break

            wav = _ensure_channels_first(wav)

            got_samples = int(wav.shape[1]) if wav.ndim == 2 else int(wav.shape[0])
            if got_samples < int(segment_samples + int(music2latent_pad_samples)):
                break

            music2latent_latents = _compute_music2latent(
                encoder=encoder,
                wav=wav,
                sample_rate=args.sample_rate,
                max_seconds=None,
            )
            latent_frames = int(music2latent_latents.shape[2])

            if isinstance(wav, np.ndarray) and wav.ndim == 2:
                wav_mono = wav.mean(axis=0)
            else:
                wav_mono = np.asarray(wav).reshape(-1)

            keep_frames = (int(latent_frames) // segment_frames) * segment_frames
            if keep_frames <= 0:
                if chunk_duration is None:
                    break
                chunk_offset = float(chunk_offset) + float(chunk_duration)
                continue

            music2latent_latents = music2latent_latents[:, :, :keep_frames]

            required_samples = int(keep_frames * samples_per_latent)
            if wav_mono.shape[0] < required_samples:
                pad = required_samples - wav_mono.shape[0]
                wav_mono = np.pad(wav_mono, (0, pad), mode="constant", constant_values=0.0)
            elif wav_mono.shape[0] > required_samples:
                wav_mono = wav_mono[:required_samples]

            num_segments = int(keep_frames // segment_frames)
            clap_tokens = torch.empty((int(num_segments), int(tokens_per_segment), 512), dtype=torch.float32)
            batch_audio: list[np.ndarray] = []
            batch_pos: list[tuple[int, int]] = []

            def _flush_batch() -> None:
                if not batch_audio:
                    return
                inputs = clap_processor(audios=batch_audio, sampling_rate=args.sample_rate, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                emb = clap_model.get_audio_features(**inputs)
                if emb.dim() > 2:
                    emb = emb.flatten(1)
                emb = emb.detach().to(dtype=torch.float32).cpu()
                if emb.dim() != 2 or emb.shape[1] != 512 or emb.shape[0] != len(batch_pos):
                    raise ValueError("Unexpected CLAP embedding batch shape")
                for j in range(int(emb.shape[0])):
                    seg_j, tok_j = batch_pos[j]
                    clap_tokens[int(seg_j), int(tok_j), :] = emb[j]
                batch_audio.clear()
                batch_pos.clear()

            for seg_idx in range(num_segments):
                seg_start = int(seg_idx * segment_samples)
                for tok_idx, start in enumerate(starts_template):
                    wstart = int(seg_start + int(start))
                    wend = int(wstart + window_len)
                    seg = wav_mono[wstart:wend]
                    if seg.shape[0] < window_len:
                        pad = int(window_len - seg.shape[0])
                        seg = np.pad(seg, (0, pad), mode="constant", constant_values=0.0)
                    batch_audio.append(seg.astype(np.float32, copy=False))
                    batch_pos.append((int(seg_idx), int(tok_idx)))
                    if len(batch_audio) >= clap_batch_size:
                        _flush_batch()

            _flush_batch()

            latents_chunks.append(music2latent_latents)
            clap_chunks.append(clap_tokens)

            if chunk_duration is None:
                break

            expected = int(round(float(args.sample_rate) * float(chunk_duration)))
            got = int(wav.shape[1]) if wav.ndim == 2 else int(wav.shape[0])

            used_seconds = float(required_samples) / float(args.sample_rate)
            if used_seconds <= 0.0:
                used_seconds = float(chunk_duration)
            chunk_offset = float(chunk_offset) + float(used_seconds)
            if total_end is None and got < int(max(1, expected - 8)):
                break

        if not latents_chunks:
            print(f"[WARN] Skipping file (too short for one segment): {os.path.basename(fpath)}")
            continue

        music2latent_latents = torch.cat(latents_chunks, dim=2)
        clap_tokens = torch.cat(clap_chunks, dim=0)
        clap_mask = torch.ones((clap_tokens.shape[0], clap_tokens.shape[1]), dtype=torch.bool)

        data = {
            "music2latent": music2latent_latents,
            "clap_tokens": clap_tokens,
            "clap_mask": clap_mask,
        }
        torch.save(data, out_path)


if __name__ == "__main__":
    main()
