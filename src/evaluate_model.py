"""Print saved evaluation metrics and show the confusion matrix."""

import json
import matplotlib.pyplot as plt
from config import METRICS_PATH, MODEL_DIR


def main():
    if not METRICS_PATH.exists():
        raise FileNotFoundError("Train the model first.")
    metrics = json.loads(METRICS_PATH.read_text())
    print("Held-out test accuracy:", round(metrics["accuracy"], 4))
    print("Training examples:", metrics["training_examples"])
    print("Testing examples:", metrics["testing_examples"])

    image_path = MODEL_DIR / "confusion_matrix.png"
    if image_path.exists():
        image = plt.imread(image_path)
        plt.figure(figsize=(12, 12))
        plt.imshow(image)
        plt.axis("off")
        plt.title("Confusion Matrix")
        plt.show()


if __name__ == "__main__":
    main()
