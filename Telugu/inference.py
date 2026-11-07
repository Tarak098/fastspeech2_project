"""
Inference script for FastSpeech 2 (Telugu).
Synthesizes speech and generates variance adaptor plots (Duration, Pitch, Energy)
for the 5 designated Telugu test sentences (T1, T2_a, T2_b, T3, T4).
"""

import os
import sys
import yaml
import torch
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

from text import symbols, text_to_sequence, telugu_to_phonemes
from model import FastSpeech2
from audio.tools import griffin_lim

TELUGU_TEST_SENTENCES = [
    ("T1", "న్యూ రోఫార్మ కోలాజికల్ పరిశోధనలోనూతన ఔషధాల ప్రభావాన్ని విశ్లషే ించారు."),
    ("T2_a", "తాడుపై అతని పట్టుచాలా బలంగా ఉంది."),
    ("T2_b", "ఆమె పట్టుచీరను ధరించి వేడుకకు వచ్చ ింది."),
    ("T3", "పులి అడ విలో నిశ్శబ్ంద గా నడిచింది."),
    ("T4", "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ంధ గా ఉంచండి.")
]

def plot_variance_adaptor(sample_id, phonemes, durations, pitch, energy, out_root):
    """
    Saves separate plots for Duration, Pitch (F0), and Energy as required by the rubric.
    """
    dur_dir = os.path.join(out_root, "variance_adaptor_plots", "duration")
    pitch_dir = os.path.join(out_root, "variance_adaptor_plots", "pitch")
    energy_dir = os.path.join(out_root, "variance_adaptor_plots", "energy")
    for d in [dur_dir, pitch_dir, energy_dir]:
        os.makedirs(d, exist_ok=True)

    durations_np = durations.squeeze(0).cpu().numpy()[:len(phonemes)]
    pitch_np = pitch.squeeze(0).cpu().numpy()
    energy_np = energy.squeeze(0).cpu().numpy()

    # 1. Duration Plot (Bar plot per phoneme)
    plt.figure(figsize=(14, 5))
    x_indices = np.arange(len(phonemes))
    plt.bar(x_indices, durations_np, color="#2ca02c", edgecolor="black", alpha=0.85)
    plt.xticks(x_indices, phonemes, rotation=75, fontsize=9)
    plt.ylabel("Predicted Duration (Frames)", fontsize=11)
    plt.title(f"FastSpeech 2 Telugu Duration Prediction - {sample_id}", fontsize=13, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(dur_dir, f"{sample_id}_duration.png"), dpi=300)
    plt.close()

    # 2. Pitch (F0) Plot
    plt.figure(figsize=(12, 4.5))
    plt.plot(pitch_np, color="#d62728", linewidth=2.0, label="Predicted Pitch ($F_0$)")
    plt.ylabel("Normalized $F_0$", fontsize=11)
    plt.xlabel("Frame Index (Hop = 256)", fontsize=11)
    plt.title(f"FastSpeech 2 Telugu Pitch ($F_0$) Contour - {sample_id}", fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(pitch_dir, f"{sample_id}_pitch.png"), dpi=300)
    plt.close()

    # 3. Energy Plot
    plt.figure(figsize=(12, 4.5))
    plt.plot(energy_np, color="#9467bd", linewidth=2.0, label="Predicted Energy")
    plt.ylabel("Normalized Energy", fontsize=11)
    plt.xlabel("Frame Index (Hop = 256)", fontsize=11)
    plt.title(f"FastSpeech 2 Telugu Energy Contour - {sample_id}", fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(energy_dir, f"{sample_id}_energy.png"), dpi=300)
    plt.close()

def synthesize(config_path="config.yaml", checkpoint_path=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Telugu Inference on {device}...")

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

    if checkpoint_path is None:
        best_ckpt = os.path.join(config["train"]["checkpoint_dir"], "best_model.pth")
        if os.path.exists(best_ckpt):
            checkpoint_path = best_ckpt
        else:
            ckpts = [os.path.join(config["train"]["checkpoint_dir"], f) for f in os.listdir(config["train"]["checkpoint_dir"]) if f.endswith(".pth")]
            checkpoint_path = ckpts[-1] if ckpts else None

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("Warning: No checkpoint found! Model running with initialized weights.")

    model.eval()

    out_root = config["train"]["results_dir"]
    audio_out_dir = os.path.join(out_root, "test_generated_audios")
    os.makedirs(audio_out_dir, exist_ok=True)

    sr = config["audio"]["sampling_rate"]

    for sample_id, text in TELUGU_TEST_SENTENCES:
        print(f"\nProcessing {sample_id}: \"{text}\"")
        phones = telugu_to_phonemes(text)
        seq = text_to_sequence(text)
        if len(seq) == 0:
            continue

        seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)

        with torch.no_grad():
            mel_before, mel_after, d_pred, p_pred, e_pred, durations = model(seq_tensor)

        plot_variance_adaptor(sample_id, phones, durations, p_pred, e_pred, out_root)

        mel_np = mel_after.squeeze(0).cpu().numpy()
        wav = griffin_lim(mel_np, sr=sr, hop_length=config["audio"]["hop_length"])

        wav_path = os.path.join(audio_out_dir, f"{sample_id}.wav")
        sf.write(wav_path, wav, sr)
        print(f"[*] Generated Telugu audio: {wav_path}")

    print(f"\nTelugu inference complete! All audios and variance plots saved in {out_root}")

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    ckpt = sys.argv[2] if len(sys.argv) > 2 else None
    synthesize(cfg, ckpt)

