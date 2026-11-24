"""Extract hand landmarks from downloaded alphabet images and save a CSV."""

from pathlib import Path
import random
import pandas as pd
from tqdm import tqdm

from config import (
    ALPHABET, IMAGE_EXTENSIONS, MAX_IMAGES_PER_CLASS,
    FEATURES_CSV, PROCESSED_DIR
)
from src.hand_detector import HandDetector
from src.feature_extractor import landmarks_to_features


def find_class_folders(search_roots):
    found = {}
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for letter in ALPHABET:
            matches = [
                p for p in root.rglob(letter)
                if p.is_dir() and p.name.upper() == letter
            ]
            for folder in matches:
                # Prefer folders that actually contain images.
                if any(f.suffix.lower() in IMAGE_EXTENSIONS for f in folder.iterdir()):
                    found.setdefault(letter, folder)
                    break
    return found


def get_search_roots():
    roots = []
    try:
        import kagglehub
        roots.append(Path(kagglehub.dataset_download("grassknoted/asl-alphabet")))
    except Exception as exc:
        print("Could not resolve KaggleHub cache:", exc)

    roots.extend([
        Path("datasets/raw"),
        Path.home() / ".cache" / "kagglehub" / "datasets"
    ])
    return roots


def main():
    folders = find_class_folders(get_search_roots())
    missing = [letter for letter in ALPHABET if letter not in folders]
    if missing:
        raise RuntimeError(
            "Could not find class folders: " + ", ".join(missing) +
            ". Run `python -m src.dataset_downloader` first and inspect the "
            "downloaded folder structure."
        )

    detector = HandDetector()
    rows = []

    try:
        for letter in ALPHABET:
            image_files = [
                p for p in folders[letter].rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            random.Random(42).shuffle(image_files)
            if MAX_IMAGES_PER_CLASS is not None:
                image_files = image_files[:MAX_IMAGES_PER_CLASS]

            print(f"\nProcessing {letter}: {len(image_files)} images")
            accepted = 0
            for image_path in tqdm(image_files):
                import cv2
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

                rows.append([letter, *features.tolist()])
                accepted += 1
            print(f"Accepted {accepted} images for {letter}")
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
