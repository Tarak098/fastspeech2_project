"""
Dataset and dynamic batch collation for FastSpeech 2 (English).
Provides on-the-fly standardization using dataset statistics and returns explicit
boolean masks for padding positions.
"""

import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np

class FastSpeech2Dataset(Dataset):
    def __init__(self, data_list_file, data_dir):
        super(FastSpeech2Dataset, self).__init__()
        self.data_dir = data_dir
        self.samples = []

        with open(data_list_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 1 and parts[0]:
                    self.samples.append(parts[0])

        # Load normalization statistics if available
        stats_path = os.path.join(data_dir, "stats.json")
        self.stats = {}
        if os.path.exists(stats_path):
            with open(stats_path, 'r', encoding='utf-8') as f:
                self.stats = json.load(f)

        self.p_mean = self.stats.get("pitch_mean", 150.0)
        self.p_std = self.stats.get("pitch_std", 50.0)
        self.e_mean = self.stats.get("energy_mean", 15.0)
        self.e_std = self.stats.get("energy_std", 10.0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        text_seq = np.load(os.path.join(self.data_dir, "text", f"{sample_id}.npy"))
        mel = np.load(os.path.join(self.data_dir, "mel", f"{sample_id}.npy"))
        duration = np.load(os.path.join(self.data_dir, "duration", f"{sample_id}.npy"))
        pitch = np.load(os.path.join(self.data_dir, "pitch", f"{sample_id}.npy"))
        energy = np.load(os.path.join(self.data_dir, "energy", f"{sample_id}.npy"))

        # Align temporal lengths
        n_frames = min(mel.shape[0], len(pitch), len(energy), int(np.sum(duration)))
        mel = mel[:n_frames]
        pitch = pitch[:n_frames]
        energy = energy[:n_frames]

        # Standardize pitch and energy to zero-mean unit-variance
        pitch_norm = (pitch - self.p_mean) / max(1e-4, self.p_std)
        energy_norm = (energy - self.e_mean) / max(1e-4, self.e_std)

        return {
            "id": sample_id,
            "text": torch.from_numpy(text_seq).long(),
            "mel": torch.from_numpy(mel).float(),
            "duration": torch.from_numpy(duration).long(),
            "pitch": torch.from_numpy(pitch_norm).float(),
            "energy": torch.from_numpy(energy_norm).float()
        }

def collate_fn(batch):
    ids = [b["id"] for b in batch]
    texts = [b["text"] for b in batch]
    mels = [b["mel"] for b in batch]
    durations = [b["duration"] for b in batch]
    pitches = [b["pitch"] for b in batch]
    energies = [b["energy"] for b in batch]

    text_lens = torch.tensor([len(t) for t in texts], dtype=torch.long)
    mel_lens = torch.tensor([m.shape[0] for m in mels], dtype=torch.long)

    max_text_len = max(text_lens).item()
    max_mel_len = max(mel_lens).item()

    padded_texts = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    padded_durations = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    padded_mels = torch.zeros(len(batch), max_mel_len, 80, dtype=torch.float)
    padded_pitches = torch.zeros(len(batch), max_mel_len, dtype=torch.float)
    padded_energies = torch.zeros(len(batch), max_mel_len, dtype=torch.float)

    for i in range(len(batch)):
        cur_t_len = text_lens[i].item()
        cur_m_len = min(mel_lens[i].item(), mels[i].shape[0])

        padded_texts[i, :cur_t_len] = texts[i][:cur_t_len]
        padded_durations[i, :cur_t_len] = durations[i][:cur_t_len]
        padded_mels[i, :cur_m_len, :] = mels[i][:cur_m_len, :]
        padded_pitches[i, :cur_m_len] = pitches[i][:cur_m_len]
        padded_energies[i, :cur_m_len] = energies[i][:cur_m_len]

    # Explicit boolean masks (True for valid positions, False for padded positions)
    src_mask = torch.arange(max_text_len).unsqueeze(0) < text_lens.unsqueeze(1)
    mel_mask = torch.arange(max_mel_len).unsqueeze(0) < mel_lens.unsqueeze(1)

    return {
        "ids": ids,
        "text": padded_texts,
        "mel": padded_mels,
        "duration": padded_durations,
        "pitch": padded_pitches,
        "energy": padded_energies,
        "text_lens": text_lens,
        "mel_lens": mel_lens,
        "src_mask": src_mask,
        "mel_mask": mel_mask
    }
