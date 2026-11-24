"""Train and save a Random Forest classifier for ASL letters and digits."""

import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

from config import (
    CLASSIFIER_PATH,
    FEATURES_CSV,
    LABELS,
    METRICS_PATH,
    MODEL_DIR,
)


def main():
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"{FEATURES_CSV} not found. Run dataset_processor first."
        )

    df = pd.read_csv(FEATURES_CSV)
    X = df.drop(columns=["label"]).values
    y = df["label"].astype(str).values

    present_labels = set(y)
    missing_labels = [label for label in LABELS if label not in present_labels]
    unexpected_labels = sorted(present_labels - set(LABELS))
    if missing_labels or unexpected_labels:
        raise ValueError(
            "Training data must contain exactly classes A-Z and 0-9. "
            f"Missing: {missing_labels or 'none'}; "
            f"unexpected: {unexpected_labels or 'none'}. "
            "Run the dataset processor again after downloading both datasets."
        )

    class_counts = pd.Series(y).value_counts()
    undersampled = [
        label for label in LABELS if class_counts.get(label, 0) < 2
    ]
    if undersampled:
        raise ValueError(
            "At least two examples are required for every class to create a "
            f"stratified train/test split. Too few examples: {undersampled}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=350,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CLASSIFIER_PATH)

    metrics = {
        "accuracy": float(accuracy),
        "training_examples": int(len(X_train)),
        "testing_examples": int(len(X_test)),
        "classification_report": report
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    cm = confusion_matrix(y_test, predictions, labels=LABELS)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    fig, ax = plt.subplots(figsize=(14, 14))
    display.plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False)
    ax.set_title("ASL Letters and Digits — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print(f"\nTest accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Saved model: {CLASSIFIER_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
