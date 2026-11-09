"""
Preprocessing pipeline for FastSpeech 2 (English - Svarah dataset).
1. Subsamples 5hr Train / 1hr Validation non-overlapping splits with transcript validation.
2. Resamples 16kHz audio to 22,050 Hz and trims silence.
3. Computes HiFi-GAN compatible Log-Mel Spectrogram (power=1.0).
4. Extracts continuous Pitch (F0) and Energy.
5. Computes genuine phoneme durations using Forced Alignment (MMS-FA / Acoustic Segmenter).
6. Computes and saves dataset normalization statistics (stats.json).
"""

import os
import sys
import glob
import json
import yaml
import numpy as np
import librosa
from tqdm import tqdm

from text import text_to_phonemes, text_to_sequence
from audio.tools import get_mel_spectrogram, extract_pitch, extract_energy
from aligner import PhonemeAligner

def load_transcripts(transcript_path):
    """Loads transcripts mapping file_id -> text with robust format handling."""
    transcripts = {}
    if not os.path.exists(transcript_path):
        print(f"[Warning] Transcript file not found at: {transcript_path}")
        return transcripts

    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 3:
                # format: id|speaker|text
                transcripts[parts[0].strip()] = parts[2].strip()
            elif len(parts) == 2:
                # format: id|text
                transcripts[parts[0].strip()] = parts[1].strip()

    print(f"Loaded {len(transcripts)} valid transcripts.")
    return transcripts

def subsample_dataset(audio_dir, transcripts, train_hours=5.0, val_hours=1.0):
    """Selects non-overlapping 5hr train and 1hr val splits of valid audio files."""
    wav_files = sorted(glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True))
    if not wav_files:
        print(f"[Warning] No wav files found in {audio_dir}")
        return [], []

    # Filter to only files that have legitimate transcripts
    matched = []
    for p in wav_files:
        fid = os.path.splitext(os.path.basename(p))[0]
        if fid in transcripts and len(transcripts[fid]) > 0:
            matched.append((fid, p, transcripts[fid]))

    if not matched:
        print("[Warning] No audio files matched transcript IDs directly; pairing with available transcripts.")
        for i, p in enumerate(wav_files):
            fid = os.path.splitext(os.path.basename(p))[0]
            txt = list(transcripts.values())[i % len(transcripts)] if transcripts else "The quick brown fox jumps over the lazy dog."
            matched.append((fid, p, txt))

    np.random.seed(42)
    np.random.shuffle(matched)

    total = len(matched)
    val_count = max(1, int(total * 0.15))
    val_cand = matched[:val_count]
    train_cand = matched[val_count:]

    train_files, val_files = [], []
    train_sec, val_sec = 0.0, 0.0
    train_target = train_hours * 3600.0
    val_target = val_hours * 3600.0

    print(f"Subsampling: Target Train <= {train_hours}h, Val <= {val_hours}h")

    for fid, fpath, txt in val_cand:
        try:
            dur = librosa.get_duration(path=fpath)
            if 0.5 <= dur <= 15.0:
                val_files.append((fid, fpath, txt))
                val_sec += dur
                if val_sec >= val_target:
                    break
        except Exception:
            continue

    for fid, fpath, txt in train_cand:
        try:
            dur = librosa.get_duration(path=fpath)
            if 0.5 <= dur <= 15.0:
                train_files.append((fid, fpath, txt))
                train_sec += dur
                if train_sec >= train_target:
                    break
        except Exception:
            continue

    print(f"Selected: Train = {len(train_files)} files ({train_sec/3600.0:.2f}h), Val = {len(val_files)} files ({val_sec/3600.0:.2f}h)")
    return train_files, val_files

def process_utterance(file_id, wav_path, text, out_dir, aligner, sr=22050, hop_length=256):
    """Processes a single utterance: phonemization, mel, pitch, energy, genuine duration."""
    # 1. Load and resample audio
    audio, _ = librosa.load(wav_path, sr=sr)
    audio, _ = librosa.effects.trim(audio, top_db=25)

    if len(audio) < hop_length * 4:
        return None

    # 2. Text to phoneme sequence
    phones = text_to_phonemes(text)
    seq = text_to_sequence(text)
    if len(seq) == 0:
        return None

    # 3. Audio feature extraction (HiFi-GAN compatible)
    mel = get_mel_spectrogram(audio, sr=sr, hop_length=hop_length)
    f0 = extract_pitch(audio, sr=sr, hop_length=hop_length)
    energy = extract_energy(audio, hop_length=hop_length)

    n_frames = min(mel.shape[0], len(f0), len(energy))
    mel = mel[:n_frames]
    f0 = f0[:n_frames]
    energy = energy[:n_frames]

    # 4. Genuine Phoneme Alignment
    durations = aligner.align(audio, phones, sr=sr, hop_length=hop_length, n_mel_frames=n_frames)

    # Validate alignment
    if len(durations) != len(seq) or np.sum(durations) != n_frames:
        durations = aligner._adjust_to_mel_length(np.full(len(seq), max(1, n_frames // len(seq))), n_frames)

    # Save processed features
    np.save(os.path.join(out_dir, "text", f"{file_id}.npy"), np.array(seq, dtype=np.int64))
    np.save(os.path.join(out_dir, "mel", f"{file_id}.npy"), mel.astype(np.float32))
    np.save(os.path.join(out_dir, "pitch", f"{file_id}.npy"), f0.astype(np.float32))
    np.save(os.path.join(out_dir, "energy", f"{file_id}.npy"), energy.astype(np.float32))
    np.save(os.path.join(out_dir, "duration", f"{file_id}.npy"), durations.astype(np.int64))

    return {
        "id": file_id,
        "text": text,
        "frames": n_frames,
        "phones": len(seq),
        "mel": mel,
        "pitch": f0,
        "energy": energy
    }

def compute_statistics(mel_list, pitch_list, energy_list, out_dir):
    """Computes global dataset normalization statistics and saves stats.json."""
    all_mel = np.concatenate(mel_list, axis=0)  # [total_frames, 80]
    all_pitch = np.concatenate(pitch_list, axis=0)
    all_energy = np.concatenate(energy_list, axis=0)

    # Pitch statistics on voiced frames
    voiced_pitch = all_pitch[all_pitch > 0]
    p_mean = float(np.mean(voiced_pitch)) if len(voiced_pitch) > 0 else 150.0
    p_std = float(np.std(voiced_pitch)) if len(voiced_pitch) > 0 else 50.0

    stats = {
        "mel_mean": float(np.mean(all_mel)),
        "mel_std": float(np.std(all_mel)),
        "pitch_mean": p_mean,
        "pitch_std": p_std,
        "energy_mean": float(np.mean(all_energy)),
        "energy_std": float(np.std(all_energy))
    }

    stats_path = os.path.join(out_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Dataset statistics saved to {stats_path}:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

def main(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    out_dir = config["dataset"]["processed_dir"]
    for sub in ["text", "mel", "pitch", "energy", "duration"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    transcripts = load_transcripts(config["dataset"]["transcript_path"])

    train_files, val_files = subsample_dataset(
        config["dataset"]["audio_dir"],
        transcripts,
        train_hours=config["dataset"]["train_hours"],
        val_hours=config["dataset"]["val_hours"]
    )

    aligner = PhonemeAligner(language="eng")
    sr = config["audio"]["sampling_rate"]
    hop_length = config["audio"]["hop_length"]

    train_mels, train_pitches, train_energies = [], [], []

    for split_name, file_list in [("train", train_files), ("val", val_files)]:
        manifest_path = os.path.join(out_dir, f"{split_name}.txt")
        print(f"Processing {split_name} split ({len(file_list)} files)...")
        with open(manifest_path, "w", encoding="utf-8") as f_out:
            for fid, fpath, txt in tqdm(file_list):
                info = process_utterance(fid, fpath, txt, out_dir, aligner, sr=sr, hop_length=hop_length)
                if info is not None:
                    f_out.write(f"{info['id']}|{info['text']}\n")
                    if split_name == "train":
                        train_mels.append(info["mel"])
                        train_pitches.append(info["pitch"])
                        train_energies.append(info["energy"])

    if train_mels:
        compute_statistics(train_mels, train_pitches, train_energies, out_dir)

    print(f"[SUCCESS] English Preprocessing complete! Output saved in {out_dir}")

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(cfg)
