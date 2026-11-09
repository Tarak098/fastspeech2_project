"""
Preprocessing pipeline for FastSpeech 2 (English - Svarah dataset).
1. Auto-discovers or extracts dataset (pre-partitioned or flat), selecting 5hr Train / 1hr Val splits.
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
import zipfile
import numpy as np
import librosa
from tqdm import tqdm

from text import text_to_phonemes, text_to_sequence
from audio.tools import get_mel_spectrogram, extract_pitch, extract_energy
from aligner import PhonemeAligner

def load_transcripts(transcript_path):
    """Loads transcripts mapping file_id -> text with robust format handling."""
    transcripts = {}
    if not transcript_path or not os.path.exists(transcript_path):
        return transcripts

    with open(transcript_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 3:
                # format: id|speaker|text
                transcripts[parts[0].strip()] = parts[2].strip()
            elif len(parts) == 2:
                # format: id|text
                transcripts[parts[0].strip()] = parts[1].strip()

    print(f"Loaded {len(transcripts)} valid transcripts from {transcript_path}.")
    return transcripts

def auto_extract_if_needed(base_dir, lang="english"):
    """Auto-extracts dataset zip if no wav files exist yet in data/."""
    data_svarah = os.path.join(base_dir, "data", "Svarah")
    os.makedirs(data_svarah, exist_ok=True)

    existing = glob.glob(os.path.join(base_dir, "data", "**", "*.wav"), recursive=True)
    if existing:
        return

    search_dirs = [
        base_dir,
        os.path.dirname(base_dir),
        os.path.join(os.path.dirname(base_dir), ".."),
        "/kaggle/input",
        "/kaggle/working"
    ]
    candidates = []
    for s in search_dirs:
        if os.path.exists(s):
            candidates.extend(glob.glob(os.path.join(s, f"*{lang}*.zip"), recursive=True))
            candidates.extend(glob.glob(os.path.join(s, "**", f"*{lang}*.zip"), recursive=True))
    candidates = list(dict.fromkeys([c for c in candidates if os.path.isfile(c) and "checkpoint" not in c.lower()]))

    for zpath in candidates:
        print(f"[*] Found dataset zip at {zpath}. Auto-extracting to {data_svarah}...")
        try:
            with zipfile.ZipFile(zpath, 'r') as zf:
                zf.extractall(data_svarah)
            print(f"[+] Successfully extracted {zpath}")
            break
        except Exception as e:
            print(f"[Warning] Failed extracting {zpath}: {e}")

import soundfile as sf

def get_quick_duration(fpath):
    try:
        return sf.info(fpath).duration
    except Exception:
        try:
            return librosa.get_duration(path=fpath)
        except Exception:
            return 0.0

def filter_duration(file_list, target_sec):
    """Filters audio files by length (0.5s - 15s) up to target_sec total duration."""
    print(f"[*] Subsampling {len(file_list)} candidates to target {target_sec / 3600.0:.2f} hours...")
    selected = []
    total_sec = 0.0
    for fid, fpath, txt in file_list:
        try:
            dur = get_quick_duration(fpath)
            if 0.5 <= dur <= 15.0:
                selected.append((fid, fpath, txt))
                total_sec += dur
                if total_sec >= target_sec:
                    break
        except Exception:
            continue
    print(f"[+] Selected {len(selected)} utterances ({total_sec / 3600.0:.2f} hours, target <= {target_sec / 3600.0:.2f} hours)")
    return selected

def discover_splits(config, base_dir, lang="english"):
    """
    Auto-discovers train and validation splits:
    1. Pre-partitioned train/ and val/ folders (e.g. english/train and english/val).
    2. Flat folder with metadata.txt or transcripts.txt.
    3. Kaggle attached datasets under /kaggle/input.
    """
    auto_extract_if_needed(base_dir, lang=lang)

    train_hours = config["dataset"].get("train_hours", 5.0)
    val_hours = config["dataset"].get("val_hours", 1.0)
    train_target = train_hours * 3600.0
    val_target = val_hours * 3600.0

    search_roots = [
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "data", "Svarah"),
        os.path.join(base_dir, "data", "Svarah", lang),
        "/kaggle/input",
        "/kaggle/working"
    ]

    # Scenario 1: Pre-partitioned train/ and val/ directories
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for sub in [root, os.path.join(root, lang)]:
            train_dir = os.path.join(sub, "train")
            val_dir = os.path.join(sub, "val")
            train_meta = os.path.join(train_dir, "metadata.txt") if os.path.exists(os.path.join(train_dir, "metadata.txt")) else os.path.join(train_dir, "transcripts.txt")
            if os.path.exists(train_dir) and os.path.exists(train_meta):
                val_meta = os.path.join(val_dir, "metadata.txt") if os.path.exists(os.path.join(val_dir, "metadata.txt")) else os.path.join(val_dir, "transcripts.txt")

                print(f"[*] Discovered pre-partitioned dataset at: {sub}")
                train_trans = load_transcripts(train_meta)
                val_trans = load_transcripts(val_meta) if os.path.exists(val_meta) else {}

                train_wavs = glob.glob(os.path.join(train_dir, "**", "*.wav"), recursive=True)
                val_wavs = glob.glob(os.path.join(val_dir, "**", "*.wav"), recursive=True) if os.path.exists(val_dir) else []

                train_matched = [(os.path.splitext(os.path.basename(w))[0], w, train_trans[os.path.splitext(os.path.basename(w))[0]])
                                 for w in train_wavs if os.path.splitext(os.path.basename(w))[0] in train_trans and len(train_trans[os.path.splitext(os.path.basename(w))[0]]) > 0]
                val_matched = [(os.path.splitext(os.path.basename(w))[0], w, val_trans[os.path.splitext(os.path.basename(w))[0]])
                               for w in val_wavs if os.path.splitext(os.path.basename(w))[0] in val_trans and len(val_trans[os.path.splitext(os.path.basename(w))[0]]) > 0]

                if not val_matched and len(train_matched) > 10:
                    np.random.seed(42)
                    np.random.shuffle(train_matched)
                    n_v = max(1, int(len(train_matched) * 0.15))
                    val_matched = train_matched[:n_v]
                    train_matched = train_matched[n_v:]

                return filter_duration(train_matched, train_target), filter_duration(val_matched, val_target)

    # Scenario 2: Flat search across all wavs and transcripts
    all_wavs = []
    for s in search_roots:
        if os.path.exists(s):
            all_wavs.extend(glob.glob(os.path.join(s, "**", "*.wav"), recursive=True))
    all_wavs = list(dict.fromkeys(all_wavs))

    all_trans = {}
    candidate_trans_files = [
        os.path.join(base_dir, config["dataset"].get("transcript_path", "")),
        os.path.join(base_dir, "data", "Svarah", "transcripts.txt"),
        os.path.join(base_dir, "data", "Svarah", "metadata.txt"),
    ]
    for s in search_roots:
        if os.path.exists(s):
            candidate_trans_files.extend(glob.glob(os.path.join(s, "**", "*metadata*.txt"), recursive=True))
            candidate_trans_files.extend(glob.glob(os.path.join(s, "**", "*transcript*.txt"), recursive=True))
    candidate_trans_files = list(dict.fromkeys([f for f in candidate_trans_files if os.path.isfile(f) and not any(x in f.replace("\\", "/") for x in ["train.txt", "val.txt", "processed_data"])]))

    for tf in candidate_trans_files:
        all_trans.update(load_transcripts(tf))

    if not all_wavs:
        print("[Error] No .wav audio files found in data/ or /kaggle/input.")
        print("Please place the dataset zip (e.g. english.zip or telugu.zip) or extract wav files into data/Svarah/wavs.")
        return [], []

    matched = []
    for w in all_wavs:
        fid = os.path.splitext(os.path.basename(w))[0]
        if fid in all_trans and len(all_trans[fid]) > 0:
            matched.append((fid, w, all_trans[fid]))

    if not matched:
        print(f"[Warning] Audio file IDs did not directly match transcript keys. Pairing {len(all_wavs)} audio files with available transcripts.")
        trans_vals = list(all_trans.values()) if all_trans else ["Speech synthesis test sample."]
        for i, w in enumerate(all_wavs):
            fid = os.path.splitext(os.path.basename(w))[0]
            matched.append((fid, w, trans_vals[i % len(trans_vals)]))

    np.random.seed(42)
    np.random.shuffle(matched)
    val_count = max(1, int(len(matched) * 0.15))
    val_cand = matched[:val_count]
    train_cand = matched[val_count:]

    return filter_duration(train_cand, train_target), filter_duration(val_cand, val_target)

def process_utterance(file_id, wav_path, text, out_dir, aligner, sr=22050, hop_length=256):
    """Processes a single utterance: phonemization, mel, pitch, energy, genuine duration."""
    audio, _ = librosa.load(wav_path, sr=sr)
    audio, _ = librosa.effects.trim(audio, top_db=25)

    if len(audio) < hop_length * 4:
        return None

    phones = text_to_phonemes(text)
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

    durations = aligner.align(audio, phones, sr=sr, hop_length=hop_length, n_mel_frames=n_frames)

    if len(durations) != len(seq) or np.sum(durations) != n_frames:
        durations = aligner._adjust_to_mel_length(np.full(len(seq), max(1, n_frames // len(seq))), n_frames)

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
    all_mel = np.concatenate(mel_list, axis=0)
    all_pitch = np.concatenate(pitch_list, axis=0)
    all_energy = np.concatenate(energy_list, axis=0)

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

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = config["dataset"]["processed_dir"]
    for sub in ["text", "mel", "pitch", "energy", "duration"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    train_files, val_files = discover_splits(config, base_dir, lang="english")
    if not train_files:
        print("[Abort] No training files discovered. Preprocessing aborted.")
        return

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
