"""
Inference script for FastSpeech 2 (Telugu).
Supports:
1. Arbitrary CLI text synthesis:
   python inference.py --text "నమస్కారం, మీరు ఎలా ఉన్నారు?" --output output_telugu.wav
2. Benchmark evaluation of the 5 prescribed test sentences (T1, T2_a, T2_b, T3, T4)
   with Variance Adaptor plots (Duration, Pitch, Energy) and neural HiFi-GAN vocoding.
"""

import os
import sys
import json
import argparse
import yaml
import torch
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

from text import symbols, text_to_sequence, telugu_to_phonemes
from model import FastSpeech2
from audio.tools import load_hifigan_vocoder, vocode_mel

TELUGU_TEST_SENTENCES = [
    ("T1", "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు."),
    ("T2_a", "తాడుపై అతని పట్టు చాలా బలంగా ఉంది."),
    ("T2_b", "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది."),
    ("T3", "పులి అడవిలో నిశ్శబ్దంగా నడిచింది."),
    ("T4", "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉండండి.")
]

def plot_variance_adaptor(sample_id, phonemes, durations, pitch, energy, out_root):
    """Saves plots for Duration, Pitch (F0), and Energy."""
    dur_dir = os.path.join(out_root, "variance_adaptor_plots", "duration")
    pitch_dir = os.path.join(out_root, "variance_adaptor_plots", "pitch")
    energy_dir = os.path.join(out_root, "variance_adaptor_plots", "energy")
    for d in [dur_dir, pitch_dir, energy_dir]:
        os.makedirs(d, exist_ok=True)

    durations_np = durations.squeeze(0).cpu().numpy()[:len(phonemes)]
    pitch_np = pitch.squeeze(0).cpu().numpy()
    energy_np = energy.squeeze(0).cpu().numpy()

    # 1. Duration Plot
    plt.figure(figsize=(12, 4))
    x_indices = np.arange(len(phonemes))
    plt.bar(x_indices, durations_np, color="#4C72B0", edgecolor="black", alpha=0.85)
    plt.xticks(x_indices, phonemes, rotation=75, fontsize=8)
    plt.ylabel("Predicted Duration (Frames)")
    plt.title(f"FastSpeech 2 Telugu Duration Prediction - {sample_id}")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(dur_dir, f"{sample_id}_duration.png"), dpi=200)
    plt.close()

    # 2. Pitch Plot
    plt.figure(figsize=(10, 3.5))
    plt.plot(pitch_np, color="#C44E52", linewidth=2.0)
    plt.ylabel("Standardized F0")
    plt.xlabel("Frame Index (Hop = 256)")
    plt.title(f"FastSpeech 2 Telugu Pitch (F0) Contour - {sample_id}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(pitch_dir, f"{sample_id}_pitch.png"), dpi=200)
    plt.close()

    # 3. Energy Plot
    plt.figure(figsize=(10, 3.5))
    plt.plot(energy_np, color="#55A868", linewidth=2.0)
    plt.ylabel("Standardized Energy")
    plt.xlabel("Frame Index (Hop = 256)")
    plt.title(f"FastSpeech 2 Telugu Energy Contour - {sample_id}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(energy_dir, f"{sample_id}_energy.png"), dpi=200)
    plt.close()

def synthesize_text(model, vocoder, text, device, stats=None, p_control=1.0, e_control=1.0, d_control=1.0):
    """Synthesizes Telugu speech waveform from input raw text string."""
    phones = telugu_to_phonemes(text)
    seq = text_to_sequence(text)
    if len(seq) == 0:
        return None, None, None, None, None

    seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)

    with torch.inference_mode():
        mel_before, mel_after, d_pred, p_pred, e_pred, durations = model(
            seq_tensor,
            p_control=p_control,
            e_control=e_control,
            d_control=d_control
        )

    p_disp = p_pred
    if stats and "pitch_mean" in stats:
        p_disp = p_pred * stats["pitch_std"] + stats["pitch_mean"]

    wav = vocode_mel(mel_after, vocoder=vocoder, device=device)

    return wav, phones, durations, p_disp, e_pred

def main():
    parser = argparse.ArgumentParser(description="FastSpeech 2 Telugu Inference")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--text", type=str, default=None, help="Raw Telugu text to synthesize")
    parser.add_argument("--output", type=str, default="generated_telugu.wav", help="Output WAV path")
    parser.add_argument("--p_control", type=float, default=1.0, help="Pitch scaling factor")
    parser.add_argument("--e_control", type=float, default=1.0, help="Energy scaling factor")
    parser.add_argument("--d_control", type=float, default=1.0, help="Duration / Speed scaling factor")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Telugu FastSpeech 2 Inference on {device}...")

    model = FastSpeech2(
        vocab_size=len(symbols),
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        d_inner=config["model"]["d_inner"],
        enc_layers=config["model"]["enc_layers"],
        dec_layers=config["model"]["dec_layers"],
        n_mels=config["audio"]["n_mels"],
        dropout=config["model"]["dropout"]
    ).to(device)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        best_ckpt = os.path.join(config["train"]["checkpoint_dir"], "best_model.pth")
        if os.path.exists(best_ckpt):
            ckpt_path = best_ckpt
        else:
            ckpts = [os.path.join(config["train"]["checkpoint_dir"], f) for f in os.listdir(config["train"]["checkpoint_dir"]) if f.endswith(".pth")]
            ckpt_path = ckpts[-1] if ckpts else None

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        try:
            try:
                state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
            except TypeError:
                state_dict = torch.load(ckpt_path, map_location=device)
            if isinstance(state_dict, dict):
                if "model" in state_dict:
                    state_dict = state_dict["model"]
                elif "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"[*] Successfully loaded checkpoint weights from {ckpt_path}.")
            if missing:
                print(f"    Missing parameters initialized: {len(missing)}")
            if unexpected:
                print(f"    Unexpected parameters ignored: {len(unexpected)}")
        except Exception as e:
            print(f"[Warning] Failed loading state_dict from {ckpt_path}: {e}")
            print("Running model in evaluation mode with current weights.")
    else:
        print("[Notice] No checkpoint found on disk. Initializing model for pipeline testing.")

    model.eval()

    print("Loading HiFi-GAN neural vocoder...")
    vocoder = load_hifigan_vocoder(device=device)
    if vocoder is not None:
        print("[*] Neural vocoder (HiFi-GAN) loaded successfully.")
    else:
        print("[*] Using high-fidelity Griffin-Lim fallback vocoder.")

    stats_path = os.path.join(config["dataset"]["processed_dir"], "stats.json")
    stats = None
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

    sr = config["audio"]["sampling_rate"]

    # 1. Single arbitrary text inference
    if args.text:
        print(f"\nSynthesizing: \"{args.text}\"")
        wav, phones, dur, p, e = synthesize_text(
            model, vocoder, args.text, device, stats,
            p_control=args.p_control, e_control=args.e_control, d_control=args.d_control
        )
        if wav is not None:
            sf.write(args.output, wav, sr)
            print(f"[SUCCESS] Telugu audio saved to {args.output} (Duration: {len(wav)/sr:.2f}s)")
        return

    # 2. Benchmark Evaluation of Prescribed Test Sentences
    out_root = config["train"]["results_dir"]
    audio_out_dir = os.path.join(out_root, "test_generated_audios")
    os.makedirs(audio_out_dir, exist_ok=True)

    print("\nRunning benchmark evaluation on the 5 Telugu test sentences (T1–T4)...")
    for sample_id, text in TELUGU_TEST_SENTENCES:
        print(f"\nProcessing {sample_id}: \"{text}\"")
        wav, phones, dur, p, e = synthesize_text(model, vocoder, text, device, stats)
        if wav is not None:
            wav_path = os.path.join(audio_out_dir, f"{sample_id}.wav")
            sf.write(wav_path, wav, sr)
            plot_variance_adaptor(sample_id, phones, dur, p, e, out_root)
            print(f"[*] Generated speech: {wav_path} ({len(wav)/sr:.2f}s)")

    print(f"\n[SUCCESS] Telugu inference complete! Results saved in {out_root}")

if __name__ == "__main__":
    main()
