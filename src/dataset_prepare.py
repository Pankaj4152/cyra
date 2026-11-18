"""
Dataset Preparation & Verification Module
==========================================
Selects 4,000 authentic grayscale binary images (2,000 benign and 2,000 malicious)
from the downloaded Maldeb dataset, removes duplicate samples using cryptographic hashes,
verifies file integrity, and creates a stratified 70:15:15 split (Train / Val / Test).
Also extracts dedicated held-out demonstration samples for offline viva presentation.
"""

import os
import shutil
import hashlib
import json
import random
from pathlib import Path
from PIL import Image

RANDOM_SEED = 42
TARGET_PER_CLASS = 2000
DEMO_SAMPLES_PER_CLASS = 6

SOURCE_DIR = Path(r"C:\Users\adity\.cache\kagglehub\datasets\saquib7hussain\maldeb-dataset\versions\1\debinew")
DEST_DIR = Path("dataset")
SAMPLES_DIR = Path("samples")


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of a file to detect duplicates."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_image(filepath: Path) -> bool:
    """Verify that an image file can be opened and is valid."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False


def prepare_dataset():
    random.seed(RANDOM_SEED)

    benign_src = SOURCE_DIR / "Benign"
    malicious_src = SOURCE_DIR / "Malicious"

    if not benign_src.exists() or not malicious_src.exists():
        raise FileNotFoundError(f"Source dataset directory not found at {SOURCE_DIR}")

    # Create destination directories
    for split in ["train", "val", "test"]:
        for cls_name in ["benign", "malicious"]:
            os.makedirs(DEST_DIR / split / cls_name, exist_ok=True)

    for cls_name in ["benign", "malicious"]:
        os.makedirs(SAMPLES_DIR / cls_name, exist_ok=True)

    print("Scanning and verifying source images...")

    def get_valid_unique_files(source_folder: Path, count_needed: int):
        all_files = [source_folder / f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(all_files)
        
        seen_hashes = set()
        valid_files = []
        
        for file_path in all_files:
            if not verify_image(file_path):
                continue
            fhash = compute_file_hash(file_path)
            if fhash in seen_hashes:
                continue
            seen_hashes.add(fhash)
            valid_files.append(file_path)
            if len(valid_files) >= count_needed:
                break
                
        return valid_files

    total_needed = TARGET_PER_CLASS + DEMO_SAMPLES_PER_CLASS
    benign_files = get_valid_unique_files(benign_src, total_needed)
    malicious_files = get_valid_unique_files(malicious_src, total_needed)

    print(f"Collected {len(benign_files)} unique, verified benign images.")
    print(f"Collected {len(malicious_files)} unique, verified malicious images.")

    # 1. Extract Held-Out Demo Samples
    demo_benign = benign_files[:DEMO_SAMPLES_PER_CLASS]
    demo_malicious = malicious_files[:DEMO_SAMPLES_PER_CLASS]

    for f in demo_benign:
        shutil.copy2(f, SAMPLES_DIR / "benign" / f.name)
    for f in demo_malicious:
        shutil.copy2(f, SAMPLES_DIR / "malicious" / f.name)

    # Remaining files for Train/Val/Test
    study_benign = benign_files[DEMO_SAMPLES_PER_CLASS:total_needed]
    study_malicious = malicious_files[DEMO_SAMPLES_PER_CLASS:total_needed]

    # 70:15:15 Split
    # 70% of 2000 = 1400
    # 15% of 2000 = 300
    # 15% of 2000 = 300
    train_n = int(TARGET_PER_CLASS * 0.70)
    val_n = int(TARGET_PER_CLASS * 0.15)
    test_n = TARGET_PER_CLASS - train_n - val_n

    splits_data = {
        "train": (study_benign[:train_n], study_malicious[:train_n]),
        "val": (study_benign[train_n:train_n + val_n], study_malicious[train_n:train_n + val_n]),
        "test": (study_benign[train_n + val_n:], study_malicious[train_n + val_n:])
    }

    manifest = {
        "dataset_name": "Maldeb Grayscale Malware & Benign Dataset",
        "total_images": TARGET_PER_CLASS * 2,
        "classes": {
            "benign": TARGET_PER_CLASS,
            "malicious": TARGET_PER_CLASS
        },
        "splits": {},
        "demo_samples": {
            "benign": len(demo_benign),
            "malicious": len(demo_malicious)
        }
    }

    for split_name, (b_list, m_list) in splits_data.items():
        print(f"Copying {len(b_list)} benign and {len(m_list)} malicious files into '{split_name}' split...")
        for bf in b_list:
            shutil.copy2(bf, DEST_DIR / split_name / "benign" / bf.name)
        for mf in m_list:
            shutil.copy2(mf, DEST_DIR / split_name / "malicious" / mf.name)
            
        manifest["splits"][split_name] = {
            "benign": len(b_list),
            "malicious": len(m_list),
            "total": len(b_list) + len(m_list)
        }

    summary_file = DEST_DIR / "dataset_summary.json"
    with open(summary_file, "w") as f:
        json.dump(manifest, f, indent=4)

    print("\nDataset preparation successfully completed!")
    print(f"Summary written to {summary_file}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    prepare_dataset()
