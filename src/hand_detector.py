"""MediaPipe Tasks wrapper: detects one hand and returns its 21 landmarks."""

from pathlib import Path
import urllib.request
import cv2
import mediapipe as mp

from config import (
    HAND_LANDMARKER_PATH, HAND_LANDMARKER_URL, NUM_HANDS,
    MIN_DETECTION_CONFIDENCE, MIN_HAND_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE
)


def ensure_hand_model():
    HAND_LANDMARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not HAND_LANDMARKER_PATH.exists():
        print("Downloading MediaPipe hand-landmarker model...")
        urllib.request.urlretrieve(HAND_LANDMARKER_URL, HAND_LANDMARKER_PATH)
    return str(HAND_LANDMARKER_PATH)


class HandDetector:
    def __init__(self, running_mode=None):
        model_path = ensure_hand_model()
        self.mp = mp
        self.mode = running_mode or mp.tasks.vision.RunningMode.IMAGE
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=self.mode,
            num_hands=NUM_HANDS,
            min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect_image(self, bgr_image):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=rgb
        )
        return self.detector.detect(mp_image)

    def detect_video(self, bgr_image, timestamp_ms):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=rgb
        )
        return self.detector.detect_for_video(mp_image, timestamp_ms)

    def close(self):
        self.detector.close()
