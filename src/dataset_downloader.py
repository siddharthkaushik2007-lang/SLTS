"""Download the ASL alphabet and digit image datasets."""

from pathlib import Path
import tempfile
import urllib.request
import zipfile

import kagglehub
from config import (
    DIGIT_DATASET_DIR,
    DIGIT_DATASET_URL,
    IMAGE_EXTENSIONS,
    KAGGLE_DATASET,
    RAW_DATA_DIR,
)


def has_digit_dataset():
    return all(
        (DIGIT_DATASET_DIR / digit).is_dir()
        and any(
            image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
            for image in (DIGIT_DATASET_DIR / digit).rglob("*")
        )
        for digit in "0123456789"
    )


def download_digit_dataset():
    if has_digit_dataset():
        print(f"Digit dataset already exists at {DIGIT_DATASET_DIR}")
        return

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=RAW_DATA_DIR) as temporary_dir:
        archive_path = Path(temporary_dir) / "sign-language-digits.zip"
        print("Downloading the sign-language digit dataset from GitHub...")
        urllib.request.urlretrieve(DIGIT_DATASET_URL, archive_path)

        extraction_root = RAW_DATA_DIR.resolve()
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target = (RAW_DATA_DIR / member.filename).resolve()
                try:
                    target.relative_to(extraction_root)
                except ValueError as exc:
                    raise ValueError(
                        f"Unsafe path in digit dataset archive: {member.filename}"
                    ) from exc
            archive.extractall(RAW_DATA_DIR)

    if not has_digit_dataset():
        raise RuntimeError(
            "The digit dataset archive did not contain all class folders 0-9."
        )
    print(f"Digit dataset downloaded to {DIGIT_DATASET_DIR}")


def main():
    print(f"Downloading {KAGGLE_DATASET} from Kaggle...")
    downloaded_path = kagglehub.dataset_download(KAGGLE_DATASET)
    print("\nDataset downloaded to:")
    print(downloaded_path)
    download_digit_dataset()


if __name__ == "__main__":
    main()
