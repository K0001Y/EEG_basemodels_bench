"""Siena Scalp EEG EDF → pickle segmentation preprocessing.

Produces the same intermediate format as CHB-MIT preprocessing:
    {'X': ndarray [16, 2560], 'y': int(0|1), 'patient': str}

Patient split:
    Train: pn01–pn10
    Val:   pn11–pn12
    Test:  pn13–pn14

Usage:
    python PEFT_engine/preprocessing/preprocess_siena.py \\
        --edf_dir "datas/Siena Scalp EEG Dataset/siena-scalp-eeg-database-1.0.0/physionet.org/files/siena-scalp-eeg/1.0.0" \\
        --output_dir datas/Siena/processed
"""

import argparse
import os
import pickle
import re

import numpy as np
import pyedflib
from tqdm import tqdm

# Standard 16 bipolar channels (same as CHB-MIT)
CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]

# Alternative channel names that may appear in Siena EDFs
CHANNEL_ALIASES = {
    "FP1-F7": ["FP1-F7", "Fp1-F7"],
    "F7-T7": ["F7-T7", "F7-T3"],
    "T7-P7": ["T7-P7", "T3-P3", "T7-P3"],
    "P7-O1": ["P7-O1", "P3-O1"],
    "FP2-F8": ["FP2-F8", "Fp2-F8"],
    "F8-T8": ["F8-T8", "F8-T4"],
    "T8-P8": ["T8-P8", "T4-P4", "T8-P4"],
    "P8-O2": ["P8-O2", "P4-O2"],
    "FP1-F3": ["FP1-F3", "Fp1-F3"],
    "F3-C3": ["F3-C3"],
    "C3-P3": ["C3-P3"],
    "P3-O1": ["P3-O1"],
    "FP2-F4": ["FP2-F4", "Fp2-F4"],
    "F4-C4": ["F4-C4"],
    "C4-P4": ["C4-P4"],
    "P4-O2": ["P4-O2"],
}

SAMPLING_RATE = 256
WINDOW_SECONDS = 10
WINDOW_SAMPLES = SAMPLING_RATE * WINDOW_SECONDS  # 2560

# Patient split
TRAIN_PATS = [f"pn{i:02d}" for i in range(1, 11)]
VAL_PATS = ["pn11", "pn12"]
TEST_PATS = ["pn13", "pn14"]


def find_channel(label: str, available: list) -> int:
    """Find a channel index by trying standard name and aliases.

    Returns:
        Index in available list, or -1 if not found.
    """
    # Direct match
    if label in available:
        return available.index(label)

    # Try aliases
    aliases = CHANNEL_ALIASES.get(label, [label])
    for alias in aliases:
        if alias in available:
            return available.index(alias)

    # Try case-insensitive match
    label_upper = label.upper()
    for i, av in enumerate(available):
        if av.upper() == label_upper:
            return i

    return -1


def read_edf_channels(edf_path: str, channels: list) -> np.ndarray:
    """Read specified channels from an EDF file with alias support.

    Args:
        edf_path: path to .edf file.
        channels: list of target channel label names.

    Returns:
        [n_channels, n_samples] array of signals.
    """
    try:
        with pyedflib.EdfReader(edf_path) as reader:
            labels = list(reader.getSignalLabels())
            n_samples = reader.getNSamples()[0]

            signal = []
            for ch in channels:
                idx = find_channel(ch, labels)
                if idx >= 0:
                    sig = reader.readSignal(idx)
                    signal.append(sig[:n_samples])
                else:
                    # Fill missing channel with zeros
                    signal.append(np.zeros(n_samples))
            return np.array(signal)
    except Exception as e:
        print(f"Error reading {edf_path}: {e}")
        return None


def parse_seizure_annotations(edf_path: str) -> list:
    """Parse seizure onset/offset from EDF annotations.

    Siena EDFs typically contain annotations as events.
    Falls back to parsing patient documentation if needed.

    Returns:
        List of (onset_sample, offset_sample) tuples.
    """
    seizure_times = []
    try:
        with pyedflib.EdfReader(edf_path) as reader:
            n_annotations = reader.read_annotations()
            # PyEDFlib's read_annotations returns a generator/list of (onset, duration, description)
            for annot in n_annotations:
                onset, duration, desc = annot[0], annot[1], annot[2]
                if "seiz" in str(desc).lower() or "onset" in str(desc).lower():
                    start_sample = int(onset * SAMPLING_RATE)
                    end_sample = int((onset + duration) * SAMPLING_RATE)
                    seizure_times.append((start_sample, end_sample))
    except Exception:
        pass

    return seizure_times


def parse_seizure_from_doc(patient_dir: str, patient: str, edf_name: str) -> list:
    """Parse seizure times from patient documentation files.

    Siena dataset has .edf.seizures files or patient info text files.

    Returns:
        List of (start_sample, end_sample) tuples.
    """
    seizure_times = []

    # Check for .seizures marker file
    seizures_file = os.path.join(patient_dir, f"{edf_name}.seizures")
    if os.path.exists(seizures_file):
        # File lists seizure start/end in seconds
        with open(seizures_file, "r") as f:
            content = f.read().strip()
            if content:
                for line in content.strip().split("\n"):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            start = int(float(parts[0]) * SAMPLING_RATE)
                            end = int(float(parts[1]) * SAMPLING_RATE)
                            seizure_times.append((start, end))
                        except ValueError:
                            pass

    return seizure_times


def segment_and_label(signal: np.ndarray, seizure_times: list,
                      patient: str, edf_name: str) -> list:
    """Split signal into 10s windows and assign seizure labels.

    Also adds oversampled seizure segments.
    """
    n_samples = signal.shape[1]
    segments = []

    # Standard 10s windows
    for i in range(0, n_samples, WINDOW_SAMPLES):
        seg = signal[:, i:i + WINDOW_SAMPLES]
        if seg.shape[1] != WINDOW_SAMPLES:
            continue

        label = 0
        for (start, end) in seizure_times:
            if i < start < i + WINDOW_SAMPLES or i < end < i + WINDOW_SAMPLES:
                label = 1
                break
            # Also check if seizure fully contains this window
            if start <= i and end >= i + WINDOW_SAMPLES:
                label = 1
                break

        segments.append({"X": seg, "y": label, "patient": patient})

    # Oversampled seizure segments
    for (start, end) in seizure_times:
        for i in range(
            max(0, start - SAMPLING_RATE),
            min(end + SAMPLING_RATE, n_samples),
            5 * SAMPLING_RATE,
        ):
            seg = signal[:, i:i + WINDOW_SAMPLES]
            if seg.shape[1] != WINDOW_SAMPLES:
                continue
            segments.append({"X": seg, "y": 1, "patient": patient})

    return segments


def process_patient(patient: str, edf_dir: str, output_dir: str):
    """Process all EDF files for one Siena patient."""
    # Find patient directory (could be pn01, PN01, etc.)
    patient_dir = None
    for d in os.listdir(edf_dir):
        if d.lower().replace("-", "") == patient.lower().replace("-", ""):
            patient_dir = os.path.join(edf_dir, d)
            break
    if patient_dir is None:
        print(f"Patient directory not found for {patient}")
        return

    # Determine output split
    if patient in TRAIN_PATS:
        split = "train"
    elif patient in VAL_PATS:
        split = "val"
    elif patient in TEST_PATS:
        split = "test"
    else:
        print(f"Unknown patient: {patient}")
        return

    out_dir = os.path.join(output_dir, split)
    os.makedirs(out_dir, exist_ok=True)

    # Process each EDF file
    edf_files = [f for f in sorted(os.listdir(patient_dir)) if f.endswith(".edf")]
    for edf_file in edf_files:
        edf_name = edf_file.replace(".edf", "")
        edf_path = os.path.join(patient_dir, edf_file)

        print(f"  Processing {edf_file}...")
        signal = read_edf_channels(edf_path, CHANNELS)
        if signal is None:
            continue

        # Parse seizure annotations
        seizure_times = parse_seizure_annotations(edf_path)
        if not seizure_times:
            seizure_times = parse_seizure_from_doc(patient_dir, patient, edf_name)

        # Segment and label
        segments = segment_and_label(signal, seizure_times, patient, edf_name)

        # Save each segment
        for seg in segments:
            base_name = f"{edf_name}_{patient}"
            counter = 0
            while os.path.exists(os.path.join(out_dir, f"{base_name}_{counter}.pkl")):
                counter += 1
            pickle.dump(seg, open(os.path.join(out_dir, f"{base_name}_{counter}.pkl"), "wb"))


def main():
    parser = argparse.ArgumentParser(description="Preprocess Siena EEG data to pickle segments")
    parser.add_argument(
        "--edf_dir", type=str,
        default="datas/Siena Scalp EEG Dataset/siena-scalp-eeg-database-1.0.0/physionet.org/files/siena-scalp-eeg/1.0.0",
        help="Root directory of Siena EEG data",
    )
    parser.add_argument("--output_dir", type=str, default="datas/Siena/processed",
                        help="Output directory for processed pickles")
    args = parser.parse_args()

    print(f"EDF directory: {args.edf_dir}")
    print(f"Output directory: {args.output_dir}")

    # Process all patients
    patients = TRAIN_PATS + VAL_PATS + TEST_PATS
    for patient in patients:
        print(f"\nProcessing {patient}...")
        process_patient(patient, args.edf_dir, args.output_dir)

    # Print summary
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(args.output_dir, split)
        if os.path.exists(split_dir):
            files = [f for f in os.listdir(split_dir) if f.endswith(".pkl")]
            print(f"{split}: {len(files)} segments")

    print("\nDone!")


if __name__ == "__main__":
    main()
