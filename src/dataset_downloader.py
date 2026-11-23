"""Download the ASL Alphabet dataset using KaggleHub's dataset API."""

import kagglehub
from config import KAGGLE_DATASET, DATASET_DIR


def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET} from Kaggle...")
    downloaded_path = kagglehub.dataset_download(KAGGLE_DATASET)
    print("\nDataset downloaded to:")
    print(downloaded_path)
    print(
        "\nYou may leave the dataset in the KaggleHub cache. "
        "The processor searches that directory automatically."
    )


if __name__ == "__main__":
    main()
