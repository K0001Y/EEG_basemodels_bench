"""CHB-MIT EDF → pickle segmentation preprocessing.

Adapts the CBraMod preprocessing pipeline (process1.py + process2.py) into
a single unified script that produces the intermediate format defined in LoRA.md §4.

Output format (per pickle):
    {'X': ndarray [16, 2560], 'y': int(0|1), 'patient': str}

Patient split:
    Train: chb01–chb20
    Val:   chb21–chb22
    Test:  chb23–chb24

Usage:
    python PEFT_engine/preprocessing/preprocess_chbmit.py \\
        --edf_dir datas/CHB-MIT \\
        --output_dir datas/CHB-MIT/processed
"""

import argparse
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pyedflib
from tqdm import tqdm

# Standard 16 bipolar channels (same as CBraMod process2.py)
CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]

SAMPLING_RATE = 256
WINDOW_SECONDS = 10
WINDOW_SAMPLES = SAMPLING_RATE * WINDOW_SECONDS  # 2560

# Patient split
TRAIN_PATS = [f"chb{i:02d}" for i in range(1, 21)]
VAL_PATS = ["chb21", "chb22"]
TEST_PATS = ["chb23", "chb24"]


def parse_summary(summary_path: str) -> dict:
    """Parse CHB-MIT summary file to extract seizure annotations.

    Returns:
        dict: filename → {'seizures': int, 'times': [(start_sample, end_sample), ...]}
    """
    with open(summary_path, "r") as f:
        lines = f.readlines()

    metadata = {}
    i = 0
    while i < len(lines):
        line = lines[i].split()
        if len(line) == 3 and line[2].endswith(".edf"):
            filename = line[2]
            j = i + 1
            # Find "Number of Seizures"
            while j < len(lines) and not lines[j].strip().startswith("Number"):
                j += 1
            if j < len(lines):
                seizures = int(lines[j].split()[-1])
            else:
                seizures = 0

            times = []
            if seizures > 0:
                j = i + 1
                for _ in range(seizures):
                    while j < len(lines):
                        l = lines[j].split()
                        if len(l) >= 3 and l[0] == "Seizure" and "Start" in lines[j]:
                            start = int(l[-2]) * SAMPLING_RATE - 1
                            # Next line has end time
                            end_line = lines[j + 1].split()
                            end = int(end_line[-2]) * SAMPLING_RATE - 1
                            times.append((start, end))
                            j += 2
                            break
                        j += 1

            metadata[filename] = {"seizures": seizures, "times": times}
        i += 1

    return metadata


def read_edf_channels(edf_path: str, channels: list) -> np.ndarray:
    """Read specified channels from an EDF file.

    Args:
        edf_path: path to .edf file.
        channels: list of channel label names.

    Returns:
        [n_channels, n_samples] array of signals.
    """
    try:
        with pyedflib.EdfReader(edf_path) as reader:
            labels = reader.getSignalLabels()
            signal = []
            for ch in channels:
                if ch in labels:
                    idx = labels.index(ch)
                    sig = reader.readSignal(idx)
                    signal.append(sig)
                else:
                    # Fill missing channel with zeros
                    n = reader.getNSamples()[0]
                    signal.append(np.zeros(n))
            return np.array(signal)
    except Exception as e:
        print(f"Error reading {edf_path}: {e}")
        return None


def segment_and_label(signal: np.ndarray, seizure_times: list,
                      patient: str, edf_name: str) -> list:
    """Split signal into 10s windows and assign seizure labels.

    Also adds oversampled seizure segments (5s shift around each seizure).

    Args:
        signal: [n_channels, n_samples] raw signal.
        seizure_times: list of (start, end) sample indices.
        patient: patient ID.
        edf_name: EDF filename (without extension).

    Returns:
        List of {'X': array, 'y': int, 'patient': str} dicts.
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

        segments.append({"X": seg, "y": label, "patient": patient})

    # Oversampled seizure segments (5s steps around each seizure)
    for idx, (start, end) in enumerate(seizure_times):
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
    """Process all EDF files for one patient.

    Args:
        patient: patient ID (e.g. 'chb01').
        edf_dir: root directory of CHB-MIT data.
        output_dir: base output directory.
    """
    patient_dir = os.path.join(edf_dir, patient)
    if not os.path.exists(patient_dir):
        print(f"Patient directory not found: {patient_dir}")
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

    # Parse summary
    summary_path = os.path.join(patient_dir, f"{patient}-summary.txt")
    if not os.path.exists(summary_path):
        print(f"Summary not found: {summary_path}")
        return
    metadata = parse_summary(summary_path)

    # Process each EDF file
    for edf_file in sorted(os.listdir(patient_dir)):
        if not edf_file.endswith(".edf"):
            continue

        edf_name = edf_file.replace(".edf", "")
        edf_path = os.path.join(patient_dir, edf_file)

        print(f"  Processing {edf_file}...")
        signal = read_edf_channels(edf_path, CHANNELS)
        if signal is None:
            continue

        # Get seizure annotations
        meta = metadata.get(edf_file, {"seizures": 0, "times": []})
        seizure_times = meta.get("times", [])

        # Segment and label
        segments = segment_and_label(signal, seizure_times, patient, edf_name)

        # Save each segment
        for seg in segments:
            fname = f"{edf_name}_{patient}.pkl"
            # Use a counter to avoid overwrites
            base_name = fname.replace(".pkl", "")
            counter = 0
            while os.path.exists(os.path.join(out_dir, f"{base_name}_{counter}.pkl")):
                counter += 1
            pickle.dump(seg, open(os.path.join(out_dir, f"{base_name}_{counter}.pkl"), "wb"))


def main():
    parser = argparse.ArgumentParser(description="Preprocess CHB-MIT data to pickle segments")
    parser.add_argument("--edf_dir", type=str, default="datas/CHB-MIT",
                        help="Root directory of CHB-MIT data")
    parser.add_argument("--output_dir", type=str, default="datas/CHB-MIT/processed",
                        help="Output directory for processed pickles")
    args = parser.parse_args()

    print(f"EDF directory: {args.edf_dir}")
    print(f"Output directory: {args.output_dir}")

    # Process all patients
    patients = [d for d in os.listdir(args.edf_dir) if d.startswith("chb")]
    patients.sort()

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
