"""Extract hand landmarks from alphabet and digit images and save a CSV."""

from pathlib import Path
import random
import cv2
import pandas as pd
from tqdm import tqdm

from config import (
    FEATURES_CSV,
    IMAGE_EXTENSIONS,
    KAGGLE_DATASET,
    LABELS,
    MAX_IMAGES_PER_CLASS,
    PROCESSED_DIR,
    RAW_DATA_DIR,
)
from src.hand_detector import HandDetector
from src.feature_extractor import landmarks_to_features


def find_class_folders(search_roots):
    found = {}
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for label in LABELS:
            matches = [
                p for p in root.rglob(label)
                if p.is_dir() and p.name.upper() == label
            ]
            for folder in matches:
                if any(
                    image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
                    for image in folder.rglob("*")
                ):
                    found.setdefault(label, folder)
                    break
    return found


def get_search_roots():
    roots = []
    try:
        import kagglehub
        roots.append(Path(kagglehub.dataset_download(KAGGLE_DATASET)))
    except Exception as exc:
        print("Could not resolve KaggleHub cache:", exc)

    roots.extend([
        RAW_DATA_DIR,
        Path.home() / ".cache" / "kagglehub" / "datasets"
    ])
    return roots


def main():
    folders = find_class_folders(get_search_roots())
    missing = [label for label in LABELS if label not in folders]
    if missing:
        raise RuntimeError(
            "Could not find image folders for classes: " + ", ".join(missing) +
            ". Run `python -m src.dataset_downloader` first and verify that the "
            "alphabet folders A-Z and digit folders 0-9 were downloaded."
        )

    detector = HandDetector()
    rows = []

    try:
        for label in LABELS:
            image_files = [
                p for p in folders[label].rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            random.Random(42).shuffle(image_files)
            if MAX_IMAGES_PER_CLASS is not None:
                image_files = image_files[:MAX_IMAGES_PER_CLASS]

            print(f"\nProcessing {label}: {len(image_files)} images")
            accepted = 0
            for image_path in tqdm(image_files):
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                result = detector.detect_image(image)
                if not result.hand_landmarks:
                    continue

                # Dataset images should contain one hand. Reject ambiguous
                # images rather than learning from an uncertain input.
                features = landmarks_to_features(result.hand_landmarks[0])
                if features is None:
                    continue

                rows.append([label, *features.tolist()])
                accepted += 1
            print(f"Accepted {accepted} images for {label}")
    finally:
        detector.close()

    if not rows:
        raise RuntimeError("No hand landmarks extracted. Check the dataset images.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["label"] + [f"f{i}" for i in range(63)]
    pd.DataFrame(rows, columns=columns).to_csv(FEATURES_CSV, index=False)
    print(f"\nSaved {len(rows)} examples to {FEATURES_CSV}")


if __name__ == "__main__":
    main()
