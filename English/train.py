"""
Training script for FastSpeech 2 (English).
Computes losses, validates, checkpoints models, and plots loss curves.
"""

import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from text import symbols
from model import FastSpeech2, FastSpeech2Loss
from dataset import FastSpeech2Dataset, collate_fn

def plot_loss(train_losses, val_losses, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Training Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Total Training Loss", color="#1f77b4", linewidth=2)
    plt.title("FastSpeech 2 English - Training Loss Curve", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_loss.png"), dpi=300)
    plt.close()

    # 2. Validation Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(val_losses, label="Total Validation Loss", color="#ff7f0e", linewidth=2)
    plt.title("FastSpeech 2 English - Validation Loss Curve", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "validation_loss.png"), dpi=300)
    plt.close()
    print(f"Loss curves saved to {out_dir}")

def train(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training FastSpeech 2 ({config['language']}) on {device}...")

    # Data loaders
    proc_dir = config["dataset"]["processed_dir"]
    train_manifest = os.path.join(proc_dir, "train.txt")
    val_manifest = os.path.join(proc_dir, "val.txt")

    if not os.path.exists(train_manifest):
        print(f"Error: Processed data not found at {train_manifest}. Run preprocess.py first!")
        return

    train_dataset = FastSpeech2Dataset(train_manifest, proc_dir)
    val_dataset = FastSpeech2Dataset(val_manifest, proc_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=1
    )

    # Initialize model and loss
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

    criterion = FastSpeech2Loss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9)

    checkpoint_dir = config["train"]["checkpoint_dir"]
    results_dir = config["train"]["results_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    epochs = config["train"]["epochs"]
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for batch in pbar:
            text = batch["text"].to(device)
            mel = batch["mel"].to(device)
            duration = batch["duration"].to(device)
            pitch = batch["pitch"].to(device)
            energy = batch["energy"].to(device)

            optimizer.zero_grad()
            mel_before, mel_after, d_pred, p_pred, e_pred, _ = model(
                text,
                duration_target=duration,
                pitch_target=pitch,
                energy_target=energy,
                max_duration_len=mel.size(1)
            )

            total_loss, m_l1, m_l2, d_l, p_l, e_l = criterion(
                mel_before, mel_after, d_pred, p_pred, e_pred,
                mel, duration, pitch, energy
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += total_loss.item()
            pbar.set_postfix({"Loss": f"{total_loss.item():.4f}", "Mel": f"{m_l2.item():.4f}"})

        epoch_train_loss = running_loss / max(1, len(train_loader))
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                text = batch["text"].to(device)
                mel = batch["mel"].to(device)
                duration = batch["duration"].to(device)
                pitch = batch["pitch"].to(device)
                energy = batch["energy"].to(device)

                mel_before, mel_after, d_pred, p_pred, e_pred, _ = model(
                    text,
                    duration_target=duration,
                    pitch_target=pitch,
                    energy_target=energy,
                    max_duration_len=mel.size(1)
                )

                v_loss, _, _, _, _, _ = criterion(
                    mel_before, mel_after, d_pred, p_pred, e_pred,
                    mel, duration, pitch, energy
                )
                running_val_loss += v_loss.item()

        epoch_val_loss = running_val_loss / max(1, len(val_loader))
        val_losses.append(epoch_val_loss)
        print(f"Epoch {epoch}: Train Loss = {epoch_train_loss:.4f} | Val Loss = {epoch_val_loss:.4f}")

        # Checkpointing
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            print(f"[*] Best model saved at epoch {epoch}")

        if epoch % config["train"]["checkpoint_interval"] == 0 or epoch == epochs:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"))

    # Plot final loss curves
    plot_loss(train_losses, val_losses, results_dir)

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    train(cfg)

