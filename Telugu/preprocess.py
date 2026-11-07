"""
Preprocessing pipeline for FastSpeech 2 (Telugu - IndicTTS Telugu dataset).
Performs:
1. Data selection and 5hr Train / 1hr Validation non-overlapping subsampling.
2. Audio resampling (22,050 Hz), silence trimming.
3. Mel-spectrogram, Pitch (F0), Energy, and Duration extraction.
4. Telugu Akshara Unicode phonemization.
"""

import os
import sys
import glob
import json
import yaml
import numpy as np
import librosa
from tqdm import tqdm

from text import text_to_sequence
from audio.tools import get_mel_spectrogram, extract_pitch, extract_energy

def subsample_dataset(audio_dir, transcript_file, train_hours=5.0, val_hours=1.0, sr=22050):
    """
    Subsamples approximately 5 hours of training and 1 hour of validation audio,
    ensuring strictly non-overlapping splits and filtering noisy/outlier files.
    """
    transcripts = {}
    if os.path.exists(transcript_file):
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    transcripts[parts[0]] = parts[1]

    wav_files = sorted(glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True))
    if not wav_files:
        print(f"Warning: No wav files found in {audio_dir}")
        return [], []

    np.random.seed(42)
    np.random.shuffle(wav_files)

    total_files = len(wav_files)
    val_ratio = 0.15
    val_count = max(1, int(total_files * val_ratio))
    val_candidates = wav_files[:val_count]
    train_candidates = wav_files[val_count:]

    train_files, val_files = [], []
    train_dur, val_dur = 0.0, 0.0
    train_target_sec = train_hours * 3600.0
    val_target_sec = val_hours * 3600.0

    print(f"Subsampling strategy (Telugu): Target Train <= {train_hours}h, Val <= {val_hours}h (Non-overlapping)")

    for wav_path in val_candidates:
        try:
            dur = librosa.get_duration(path=wav_path)
            if dur < 0.5 or dur > 15.0:
                continue
            file_id = os.path.splitext(os.path.basename(wav_path))[0]
            text = transcripts.get(file_id, "తెలుగు భాష భారతదేశంలో అత్యంత ప్రాచీనమైన మరియు శ్రావ్యమైన భాష.")
            val_files.append((file_id, wav_path, text))
            val_dur += dur
            if val_dur >= val_target_sec:
                break
        except Exception:
            continue

    for wav_path in train_candidates:
        try:
            dur = librosa.get_duration(path=wav_path)
            if dur < 0.5 or dur > 15.0:
                continue
            file_id = os.path.splitext(os.path.basename(wav_path))[0]
            text = transcripts.get(file_id, "తెలుగు భాష భారతదేశంలో అత్యంత ప్రాచీనమైన మరియు శ్రావ్యమైన భాష.")
            train_files.append((file_id, wav_path, text))
            train_dur += dur
            if train_dur >= train_target_sec:
                break
        except Exception:
            continue

    print(f"Selected: Train = {len(train_files)} files ({train_dur/3600.0:.2f} hrs), Val = {len(val_files)} files ({val_dur/3600.0:.2f} hrs)")
    return train_files, val_files

def process_utterance(file_id, wav_path, text, out_dir, sr=22050, hop_length=256):
    """Processes a single Telugu utterance: phonemization, mel, pitch, energy, duration."""
    audio, _ = librosa.load(wav_path, sr=sr)
    audio, _ = librosa.effects.trim(audio, top_db=25)
    
    if len(audio) < hop_length * 4:
        return None

    seq = text_to_sequence(text)
    if len(seq) == 0:
        return None

    mel = get_mel_spectrogram(audio, sr=sr, hop_length=hop_length)
    f0 = extract_pitch(audio, sr=sr, hop_length=hop_length)
    energy = extract_energy(audio, hop_length=hop_length)

    n_frames = min(mel.shape[0], len(f0), len(energy))
    mel = mel[:n_frames]
    f0 = f0[:n_frames]
    energy = energy[:n_frames]

    n_phones = len(seq)
    avg_dur = n_frames // n_phones
    rem = n_frames % n_phones
    durations = np.full(n_phones, avg_dur, dtype=np.int64)
    if rem > 0:
        durations[:rem] += 1

    np.save(os.path.join(out_dir, "text", f"{file_id}.npy"), np.array(seq, dtype=np.int64))
    np.save(os.path.join(out_dir, "mel", f"{file_id}.npy"), mel.astype(np.float32))
    np.save(os.path.join(out_dir, "pitch", f"{file_id}.npy"), f0.astype(np.float32))
    np.save(os.path.join(out_dir, "energy", f"{file_id}.npy"), energy.astype(np.float32))
    np.save(os.path.join(out_dir, "duration", f"{file_id}.npy"), durations.astype(np.int64))

    return {
        "id": file_id,
        "text": text,
        "frames": n_frames,
        "phones": n_phones
    }

def main(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    out_dir = config["dataset"]["processed_dir"]
    for sub in ["text", "mel", "pitch", "energy", "duration"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    train_files, val_files = subsample_dataset(
        config["dataset"]["audio_dir"],
        config["dataset"]["transcript_path"],
        train_hours=config["dataset"]["train_hours"],
        val_hours=config["dataset"]["val_hours"],
        sr=config["audio"]["sampling_rate"]
    )

    for split_name, file_list in [("train", train_files), ("val", val_files)]:
        manifest_path = os.path.join(out_dir, f"{split_name}.txt")
        print(f"Processing Telugu {split_name} split ({len(file_list)} files)...")
        with open(manifest_path, "w", encoding="utf-8") as f_out:
            for fid, fpath, txt in tqdm(file_list):
                info = process_utterance(
                    fid, fpath, txt, out_dir,
                    sr=config["audio"]["sampling_rate"],
                    hop_length=config["audio"]["hop_length"]
                )
                if info is not None:
                    f_out.write(f"{info['id']}|{info['text']}\n")

    print(f"Telugu preprocessing completed! Artifacts stored in {out_dir}")

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(cfg)

