from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_DIR = Path.home() / ".cache" / "kagglehub" / "datasets" / "grassknoted" / "asl-alphabet" / "versions" / "1" / "asl_alphabet_train" / "asl_alphabet_train"
PROCESSED_DIR = ROOT / "datasets" / "processed"
MODEL_DIR = ROOT / "models"

# Kaggle dataset selected for the first ASL alphabet baseline.
KAGGLE_DATASET = "grassknoted/asl-alphabet"

# Limit images per class to keep landmark extraction manageable.
# Set to None to process every image (can take a long time).
MAX_IMAGES_PER_CLASS = 1500

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_LANDMARKER_PATH = MODEL_DIR / "hand_landmarker.task"
CLASSIFIER_PATH = MODEL_DIR / "asl_alphabet_rf.joblib"
METRICS_PATH = MODEL_DIR / "training_metrics.json"
FEATURES_CSV = PROCESSED_DIR / "asl_landmarks.csv"

NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
PREDICTION_THRESHOLD = 0.55
SMOOTHING_WINDOW = 9
