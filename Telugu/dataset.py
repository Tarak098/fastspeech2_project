"""
Dataset and dynamic batch collation for FastSpeech 2 (Telugu).
"""

import os
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
                if len(parts) >= 2:
                    sample_id = parts[0]
                    self.samples.append(sample_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        text_seq = np.load(os.path.join(self.data_dir, "text", f"{sample_id}.npy"))
        mel = np.load(os.path.join(self.data_dir, "mel", f"{sample_id}.npy"))
        duration = np.load(os.path.join(self.data_dir, "duration", f"{sample_id}.npy"))
        pitch = np.load(os.path.join(self.data_dir, "pitch", f"{sample_id}.npy"))
        energy = np.load(os.path.join(self.data_dir, "energy", f"{sample_id}.npy"))
        
        return {
            "id": sample_id,
            "text": torch.from_numpy(text_seq).long(),
            "mel": torch.from_numpy(mel).float(),
            "duration": torch.from_numpy(duration).long(),
            "pitch": torch.from_numpy(pitch).float(),
            "energy": torch.from_numpy(energy).float()
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
        padded_texts[i, :text_lens[i]] = texts[i]
        padded_durations[i, :text_lens[i]] = durations[i]
        
        cur_mel_len = min(mel_lens[i].item(), mels[i].shape[0])
        padded_mels[i, :cur_mel_len, :] = mels[i][:cur_mel_len, :]
        padded_pitches[i, :cur_mel_len] = pitches[i][:cur_mel_len]
        padded_energies[i, :cur_mel_len] = energies[i][:cur_mel_len]

    return {
        "ids": ids,
        "text": padded_texts,
        "mel": padded_mels,
        "duration": padded_durations,
        "pitch": padded_pitches,
        "energy": padded_energies,
        "text_lens": text_lens,
        "mel_lens": mel_lens
    }

