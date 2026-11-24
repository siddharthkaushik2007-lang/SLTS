from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT / "datasets" / "raw"
DIGIT_DATASET_DIR = (
    RAW_DATA_DIR / "Sign-Language-Digits-Dataset-master" / "Dataset"
)
PROCESSED_DIR = ROOT / "datasets" / "processed"
MODEL_DIR = ROOT / "models"

# Kaggle datasets used for the ASL alphabet and digit classes.
KAGGLE_DATASET = "grassknoted/asl-alphabet"
DIGIT_DATASET_URL = (
    "https://codeload.github.com/ardamavi/Sign-Language-Digits-Dataset/"
    "zip/refs/heads/master"
)

# Limit images per class to keep landmark extraction manageable.
# Set to None to process every image (can take a long time).
MAX_IMAGES_PER_CLASS = 1500

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NUMERALS = list("0123456789")
LABELS = ALPHABET + NUMERALS

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_LANDMARKER_PATH = MODEL_DIR / "hand_landmarker.task"
CLASSIFIER_PATH = MODEL_DIR / "asl_alphanumeric_rf.joblib"
METRICS_PATH = MODEL_DIR / "training_metrics.json"
FEATURES_CSV = PROCESSED_DIR / "asl_alphanumeric_landmarks.csv"

NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
PREDICTION_THRESHOLD = 0.55
SMOOTHING_WINDOW = 9
