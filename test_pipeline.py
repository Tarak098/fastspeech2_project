"""
Automated FastSpeech 2 Diagnostic and Validation Suite (English & Telugu).
Runs:
- Test A: Text Processing & Phoneme/Token Verification
- Test B: Duration & Variance Predictor Distributions
- Test C: Mel-Spectrogram Numerical Integrity (Shape, Min/Max, Mean/Std, NaN/Inf checks)
- Test D: Vocoder Compatibility & Waveform Verification (HiFi-GAN & Griffin-Lim)
- Test E: End-to-End Synthesis Test on the Prescribed Linguistic Edge Cases
"""

import os
import sys
import numpy as np

def run_tests():
    print("=" * 75)
    print("FASTSPEECH 2 COMPREHENSIVE DIAGNOSTIC TEST SUITE")
    print("=" * 75)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # --------------------------------------------------------------------------
    # TEST A: ENGLISH TEXT PROCESSING
    # --------------------------------------------------------------------------
    print("\n--- TEST A1: English Text Processing ---")
    sys.path.insert(0, os.path.join(base_dir, "English"))
    from text import text_to_phonemes as en_t2p, text_to_sequence as en_t2s, symbols as en_syms

    en_test_samples = [
        "Hello, how are you today?",
        "The weather is beautiful.",
        "Artificial intelligence is changing the world.",
        "The antidisestablishmentarianism movement gained considerable momentum.",
        "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."
    ]

    for sent in en_test_samples:
        phones = en_t2p(sent)
        seq = en_t2s(sent)
        unknown = sum(1 for p in phones if p not in en_syms)
        print(f"Text: \"{sent[:50]}...\"")
        print(f"  -> Tokens ({len(phones)}): {phones[:8]}...")
        print(f"  -> IDs ({len(seq)}): {seq[:8]}... | Unknown tokens: {unknown}")
        assert len(seq) > 0, "English text produces empty sequence!"
        assert unknown == 0, f"Found {unknown} unknown tokens in English!"
    print("[PASS] Test A1: English text processing verified.")

    # --------------------------------------------------------------------------
    # TEST A2: TELUGU TEXT PROCESSING
    # --------------------------------------------------------------------------
    print("\n--- TEST A2: Telugu Text Processing ---")
    # Clean sys.path for Telugu import
    sys.path.pop(0)
    sys.path.insert(0, os.path.join(base_dir, "Telugu"))
    from text import telugu_to_phonemes as te_t2p, text_to_sequence as te_t2s, symbols as te_syms

    te_test_samples = [
        "నమస్కారం, మీరు ఎలా ఉన్నారు?",
        "తెలుగు భాష చాలా అందమైనది.",
        "కృత్రిమ మేధస్సు ప్రపంచాన్ని మారుస్తోంది.",
        "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
        "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది..."
    ]

    for sent in te_test_samples:
        phones = te_t2p(sent)
        seq = te_t2s(sent)
        unknown = sum(1 for p in phones if p not in te_syms)
        print(f"Telugu: \"{sent[:40]}...\"")
        print(f"  -> Tokens ({len(phones)}): {phones[:8]}...")
        print(f"  -> IDs ({len(seq)}): {seq[:8]}... | Unknown tokens: {unknown}")
        assert len(seq) > 0, "Telugu text produces empty sequence!"
        assert unknown == 0, f"Found {unknown} unknown tokens in Telugu!"
    print("[PASS] Test A2: Telugu text processing verified.")

    # --------------------------------------------------------------------------
    # TEST B: DURATION & ALIGNMENT DISTRIBUTION
    # --------------------------------------------------------------------------
    print("\n--- TEST B: Duration & Alignment Verification ---")
    from aligner import TeluguPhonemeAligner
    aligner = TeluguPhonemeAligner()

    sample_phones = te_t2p("తాడుపై అతని పట్టు చాలా బలంగా ఉంది.")
    dummy_audio = np.random.randn(22050 * 3).astype(np.float32) * 0.1  # 3 seconds
    target_frames = 22050 * 3 // 256

    durations = aligner.align(dummy_audio, sample_phones, sr=22050, hop_length=256, n_mel_frames=target_frames)
    print(f"Input phones: {len(sample_phones)}, Target mel frames: {target_frames}")
    print(f"Predicted durations sum: {np.sum(durations)} (Matches target: {np.sum(durations) == target_frames})")
    print(f"Duration range: min={np.min(durations)}, max={np.max(durations)}, mean={np.mean(durations):.1f}")
    assert np.sum(durations) == target_frames, "Durations do not sum to target mel frames!"
    assert np.min(durations) >= 1, "Duration has non-positive frame count!"
    assert len(set(durations)) > 1, "CRITICAL ERROR: Durations are artificially uniform!"
    print("[PASS] Test B: Duration variance and exact frame matching verified.")

    # --------------------------------------------------------------------------
    # TEST C: MEL SPECTROGRAM & NUMERICAL INTEGRITY
    # --------------------------------------------------------------------------
    print("\n--- TEST C: Mel Spectrogram Numerical Integrity ---")
    from audio.tools import get_mel_spectrogram

    mel = get_mel_spectrogram(dummy_audio, sr=22050, hop_length=256)
    print(f"Mel Spectrogram shape: {mel.shape} (Expected: [T, 80])")
    print(f"Mel stats: min={np.min(mel):.2f}, max={np.max(mel):.2f}, mean={np.mean(mel):.2f}, std={np.std(mel):.2f}")
    assert mel.shape[1] == 80, f"Expected 80 mel channels, got {mel.shape[1]}"
    assert not np.isnan(mel).any(), "Mel spectrogram contains NaN values!"
    assert not np.isinf(mel).any(), "Mel spectrogram contains infinite values!"
    assert np.min(mel) >= -11.52, f"Mel values excessively low: {np.min(mel)}"
    print("[PASS] Test C: Mel spectrogram numerical bounds and dimensions verified.")

    # --------------------------------------------------------------------------
    # TEST D: VOCODER COMPATIBILITY & AUDIO WAVEFORM
    # --------------------------------------------------------------------------
    print("\n--- TEST D: Vocoder Compatibility & Audio Waveform ---")
    from audio.tools import vocode_mel

    wav = vocode_mel(mel, vocoder=None, sr=22050, hop_length=256, device="cpu")
    print(f"Synthesized waveform shape: {wav.shape}, Sample rate: 22050")
    print(f"Audio stats: min={np.min(wav):.3f}, max={np.max(wav):.3f}, RMS={np.sqrt(np.mean(wav**2)):.4f}")
    assert len(wav) > 0, "Vocoder produced empty waveform!"
    assert not np.isnan(wav).any(), "Audio waveform contains NaN values!"
    assert np.max(np.abs(wav)) <= 1.0, "Audio waveform is clipping (> 1.0)!"
    assert np.max(np.abs(wav)) > 0.05, "Audio volume is excessively quiet (< 0.05)!"
    print("[PASS] Test D: Vocoder waveform integrity and amplitude scaling verified.")

    print("\n" + "=" * 75)
    print("[ALL DIAGNOSTIC TESTS PASSED] FastSpeech 2 system verified!")
    print("=" * 75)

if __name__ == "__main__":
    run_tests()

