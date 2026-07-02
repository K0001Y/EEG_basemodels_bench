"""TUSZ v2.0.6 EDF → pickle segmentation preprocessing.

Produces the same intermediate format as CHB-MIT / Siena preprocessing:
    {'X': ndarray [16, 2560], 'y': int(0|1), 'patient': str}

TUSZ specifics:
    - Sampling rates vary: 250 Hz (LE) or 400 Hz (AR); resampled to 256 Hz
    - Channels are referential (EEG FP1-LE, EEG F7-REF, etc.)
    - 16 bipolar derivations computed from referential channels
    - Binary annotations: .csv_bi files (seiz / bckg)
    - Split: train / dev / eval (mapped to train / val / test)

Usage:
    python PEFT_engine/preprocessing/preprocess_tusz.py \\
        --edf_dir datas/tusz_v2.0.6/edf \\
        --output_dir datas/TUSZ/processed
"""

import argparse
import os
import pickle
import csv

import numpy as np
import pyedflib
from scipy import signal as scipy_signal
from tqdm import tqdm

# Target sampling rate for intermediate format
TARGET_SR = 256
WINDOW_SECONDS = 10
WINDOW_SAMPLES = TARGET_SR * WINDOW_SECONDS  # 2560

# Referential electrode names that may appear in TUSZ EDF files
REF_SUFFIXES = ["-LE", "-REF"]

# 16 bipolar derivations: each is (pos_electrode, neg_electrode)
# Matches CHB-MIT standard channel order
BIPOLAR_PAIRS = [
    ("FP1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
    ("FP2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
]

# TUSZ uses old nomenclature: T3=T7, T4=T8, T5=P7, T6=P8
ALIAS_MAP = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

# Split mapping: TUSZ directory name → our split name
SPLIT_MAP = {"train": "train", "dev": "val", "eval": "test"}


def find_ref_channel(label: str, available: dict) -> int:
    """Find a referential channel index by trying all suffix variants.

    Args:
        label: electrode name without suffix, e.g. 'FP1'.
        available: dict mapping channel_label -> index.

    Returns:
        Channel index, or -1 if not found.
    """
    # Apply alias (T3→T7, etc.)
    resolved = ALIAS_MAP.get(label, label)

    for suffix in REF_SUFFIXES:
        for prefix in ["EEG ", ""]:
            candidate = f"{prefix}{resolved}{suffix}"
            if candidate in available:
                return available[candidate]

    # Case-insensitive fallback
    target = resolved.upper()
    for ch_label, idx in available.items():
        # Extract electrode name: "EEG FP1-LE" → "FP1"
        parts = ch_label.replace("EEG ", "").split("-")
        if parts and parts[0].upper() == target:
            return idx

    return -1


def read_referential_channels(edf_path: str) -> tuple:
    """Read referential channels from TUSZ EDF and compute 16 bipolar derivations.

    Returns:
        (signal, sampling_rate): [16, n_samples] bipolar signal + original sampling rate,
        or (None, None) on failure.
    """
    try:
        with pyedflib.EdfReader(edf_path) as reader:
            labels = reader.getSignalLabels()
            available = {lbl: i for i, lbl in enumerate(labels)}
            sr = reader.getSampleFrequency(0)

            # Build electrode → signal cache
            electrode_cache = {}
            for (pos, neg) in BIPOLAR_PAIRS:
                for elec in [pos, neg]:
                    resolved = ALIAS_MAP.get(elec, elec)
                    if resolved not in electrode_cache:
                        idx = find_ref_channel(elec, available)
                        if idx >= 0:
                            electrode_cache[resolved] = reader.readSignal(idx)
                        else:
                            # Missing electrode → zeros
                            n_samples = reader.getNSamples()[0]
                            electrode_cache[resolved] = np.zeros(n_samples)

            # Compute bipolar derivations
            bipolar_signals = []
            for (pos, neg) in BIPOLAR_PAIRS:
                pos_resolved = ALIAS_MAP.get(pos, pos)
                neg_resolved = ALIAS_MAP.get(neg, neg)
                derivation = electrode_cache[pos_resolved] - electrode_cache[neg_resolved]
                bipolar_signals.append(derivation)

            signal = np.array(bipolar_signals)  # [16, n_samples]
            return signal, sr

    except Exception as e:
        print(f"Error reading {edf_path}: {e}")
        return None, None


def parse_csv_bi(csv_path: str) -> list:
    """Parse TUSZ .csv_bi annotation file.

    Returns:
        List of (start_sec, end_sec) tuples for seizure events.
    """
    seizure_times = []
    if not os.path.exists(csv_path):
        return seizure_times

    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line.startswith("channel") or not line:
                continue
            parts = line.split(",")
            if len(parts) >= 4:
                label = parts[3].strip().lower()
                if "seiz" in label:
                    start = float(parts[1])
                    end = float(parts[2])
                    seizure_times.append((start, end))

    return seizure_times


def resample_to_target(signal: np.ndarray, orig_sr: float) -> np.ndarray:
    """Resample signal from original sampling rate to TARGET_SR (256 Hz).

    Args:
        signal: [n_channels, n_samples] at original sampling rate.

    Returns:
        [n_channels, n_samples_resampled] at 256 Hz.
    """
    if orig_sr == TARGET_SR:
        return signal

    n_samples_new = int(signal.shape[1] * TARGET_SR / orig_sr)
    return scipy_signal.resample(signal, n_samples_new, axis=1)


def segment_and_label(signal: np.ndarray, seizure_times: list,
                      patient: str, sr: float = TARGET_SR) -> list:
    """Split resampled signal into windows and assign seizure labels.

    Args:
        signal: [16, n_samples] at TARGET_SR.
        seizure_times: list of (start_sec, end_sec).
        patient: patient ID.
        sr: sampling rate (should be TARGET_SR after resampling).

    Returns:
        List of {'X': array, 'y': int, 'patient': str} dicts.
    """
    window_samples = int(sr * WINDOW_SECONDS)
    n_samples = signal.shape[1]
    segments = []

    # Standard windows
    for i in range(0, n_samples, window_samples):
        seg = signal[:, i:i + window_samples]
        if seg.shape[1] != window_samples:
            continue

        start_sec = i / sr
        end_sec = (i + window_samples) / sr

        label = 0
        for (sz_start, sz_end) in seizure_times:
            # Overlap check: window overlaps with seizure
            if start_sec < sz_end and end_sec > sz_start:
                label = 1
                break

        segments.append({"X": seg, "y": label, "patient": patient})

    # Oversampled seizure segments (5s steps around each seizure)
    step = int(5 * sr)
    for (sz_start, sz_end) in seizure_times:
        start_sample = max(0, int(sz_start * sr) - int(sr))
        end_sample = min(n_samples, int(sz_end * sr) + int(sr))
        for i in range(start_sample, end_sample, step):
            seg = signal[:, i:i + window_samples]
            if seg.shape[1] != window_samples:
                continue
            segments.append({"X": seg, "y": 1, "patient": patient})

    return segments


def process_split(split_name: str, split_dir: str, output_dir: str):
    """Process all patients in one split (train/dev/eval).

    Args:
        split_name: 'train', 'dev', or 'eval'.
        split_dir: path to the split directory.
        output_dir: base output directory.
    """
    our_split = SPLIT_MAP.get(split_name, split_name)
    out_dir = os.path.join(output_dir, our_split)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(split_dir):
        print(f"Split directory not found: {split_dir}")
        return

    patients = sorted([d for d in os.listdir(split_dir)
                       if os.path.isdir(os.path.join(split_dir, d))])

    total_segments = 0
    total_seizure = 0

    for patient in tqdm(patients, desc=f"TUSZ {split_name}"):
        patient_dir = os.path.join(split_dir, patient)

        # Walk through sessions and montages
        for session in sorted(os.listdir(patient_dir)):
            session_dir = os.path.join(patient_dir, session)
            if not os.path.isdir(session_dir):
                continue

            for montage in sorted(os.listdir(session_dir)):
                montage_dir = os.path.join(session_dir, montage)
                if not os.path.isdir(montage_dir):
                    continue

                # Find all EDF files in this montage directory
                edf_files = sorted([f for f in os.listdir(montage_dir)
                                    if f.endswith(".edf")])

                for edf_file in edf_files:
                    edf_path = os.path.join(montage_dir, edf_file)
                    base_name = edf_file.replace(".edf", "")

                    # Parse annotations
                    csv_bi_path = edf_path.replace(".edf", ".csv_bi")
                    if not os.path.exists(csv_bi_path):
                        csv_bi_path = edf_path.replace(".edf", ".csv")
                        if not os.path.exists(csv_bi_path):
                            continue

                    # Prefer .csv_bi for binary labels
                    seizure_times = parse_csv_bi(
                        edf_path.replace(".edf", ".csv_bi"))
                    if not seizure_times:
                        # Fall back to .csv — any seizure-type label counts
                        seizure_times = _parse_csv_seizures(
                            edf_path.replace(".edf", ".csv"))

                    # Read and compute bipolar channels
                    signal, sr = read_referential_channels(edf_path)
                    if signal is None:
                        continue

                    # Resample to 256 Hz
                    signal = resample_to_target(signal, sr)

                    # Segment and label
                    segments = segment_and_label(signal, seizure_times, patient)

                    # Save
                    for seg in segments:
                        counter = 0
                        while os.path.exists(
                            os.path.join(out_dir,
                                         f"{base_name}_{counter}.pkl")):
                            counter += 1
                        fname = f"{base_name}_{counter}.pkl"
                        with open(os.path.join(out_dir, fname), "wb") as f:
                            pickle.dump(seg, f)
                        total_segments += 1
                        if seg["y"] == 1:
                            total_seizure += 1

                    # Only process one montage per file (avoid duplicates)
                    # If multiple montages exist, we take the first one found
                break  # Only use first montage per session

    print(f"  {split_name} → {our_split}: {total_segments} segments "
          f"({total_seizure} seizure, "
          f"{total_segments - total_seizure} background)")


def _parse_csv_seizures(csv_path: str) -> list:
    """Parse TUSZ .csv (multi-class) for any seizure-type labels.

    Returns:
        List of (start_sec, end_sec) tuples.
    """
    seizure_times = []
    if not os.path.exists(csv_path):
        return seizure_times

    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line.startswith("channel") or not line:
                continue
            parts = line.split(",")
            if len(parts) >= 4:
                label = parts[3].strip().lower()
                # Any seizure type (cpsz, gnsz, fnsz, tnSz, etc.) → positive
                if "sz" in label and label != "bckg":
                    start = float(parts[1])
                    end = float(parts[2])
                    seizure_times.append((start, end))

    # Merge overlapping intervals
    if not seizure_times:
        return seizure_times
    seizure_times.sort()
    merged = [seizure_times[0]]
    for start, end in seizure_times[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess TUSZ v2.0.6 data to pickle segments")
    parser.add_argument("--edf_dir", type=str,
                        default="datas/tusz_v2.0.6/edf",
                        help="Root directory containing train/dev/eval subdirs")
    parser.add_argument("--output_dir", type=str,
                        default="datas/TUSZ/processed",
                        help="Output directory for processed pickles")
    args = parser.parse_args()

    print(f"EDF directory: {args.edf_dir}")
    print(f"Output directory: {args.output_dir}")

    for split_name in ["train", "dev", "eval"]:
        split_dir = os.path.join(args.edf_dir, split_name)
        if os.path.isdir(split_dir):
            print(f"\nProcessing {split_name}...")
            process_split(split_name, split_dir, args.output_dir)

    # Print summary
    print("\n" + "=" * 50)
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(args.output_dir, split)
        if os.path.exists(split_dir):
            files = [f for f in os.listdir(split_dir) if f.endswith(".pkl")]
            print(f"{split}: {len(files)} segments")
    print("\nDone!")


if __name__ == "__main__":
    main()
