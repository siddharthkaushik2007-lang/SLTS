"""Live one-hand ASL alphabet recognition using the trained model."""
from src.tts import TextToSpeech
from collections import deque, Counter
import time

import cv2
import joblib

from config import CLASSIFIER_PATH, SMOOTHING_WINDOW
from src.hand_detector import HandDetector
from src.feature_extractor import landmarks_to_features, draw_hand_landmarks


# Show a prediction only when the highest class probability is at least 20%.
MIN_DISPLAY_PROBABILITY = 0.20
TOP_K = 3


def main():
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run the downloader, processor and trainer first."
        )

    model = joblib.load(CLASSIFIER_PATH)
    detector = HandDetector(running_mode=detector_mode())

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        detector.close()
        raise RuntimeError("Could not open webcam. Check camera permissions.")

    history = deque(maxlen=SMOOTHING_WINDOW)
    current_letter = "—"
    top_predictions = []
    last_timestamp = 0
    tts = TextToSpeech()
    last_spoken_letter = None
    print("SLTS running. Press Q to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame from the webcam.")
                break

            # Mirror preview for a natural webcam experience.
            frame = cv2.flip(frame, 1)
            timestamp = int(time.monotonic() * 1000)
            timestamp = max(timestamp, last_timestamp + 1)
            last_timestamp = timestamp

            result = detector.detect_video(frame, timestamp)

            if result.hand_landmarks:
                # num_hands=1 ensures the detector returns at most one hand.
                hand = result.hand_landmarks[0]
                frame = draw_hand_landmarks(frame, hand)
                features = landmarks_to_features(hand)

                if features is not None:
                    probabilities = model.predict_proba([features])[0]
                    best_index = int(probabilities.argmax())
                    best_probability = float(probabilities[best_index])

                    if best_probability >= MIN_DISPLAY_PROBABILITY:
                        prediction = str(model.classes_[best_index])

                        # Smooth the main displayed letter across recent frames.
                        history.append(prediction)
                        current_letter = Counter(history).most_common(1)[0][0]
                        if current_letter != last_spoken_letter:
                         tts.speak(current_letter)
                         last_spoken_letter = current_letter

                        # Show the three most likely classes for the current frame.
                        top_indices = probabilities.argsort()[-TOP_K:][::-1]
                        top_predictions = [
                            (str(model.classes_[i]), float(probabilities[i]))
                            for i in top_indices
                        ]
                    else:
                        # Below 20%: show no predicted letter or top-three list.
                        history.clear()
                        current_letter = "—"
                        top_predictions = []
                else:
                    history.clear()
                    current_letter = "—"
                    top_predictions = []
            else:
                history.clear()
                current_letter = "Show one hand"
                top_predictions = []

            # Prediction panel.
            cv2.rectangle(frame, (10, 10), (410, 185), (20, 20, 20), -1)
            cv2.putText(
                frame,
                f"Letter: {current_letter}",
                (25, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (80, 240, 120),
                2,
            )

            if top_predictions:
                cv2.putText(
                    frame,
                    "Top 3 predictions:",
                    (25, 82),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (240, 240, 240),
                    1,
                )

                for rank, (letter, probability) in enumerate(top_predictions, start=1):
                    y = 108 + (rank - 1) * 24
                    cv2.putText(
                        frame,
                        f"{rank}. {letter}  ({probability * 100:.1f}%)",
                        (35, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (240, 240, 240),
                        1,
                    )
            elif current_letter == "—":
                cv2.putText(
                    frame,
                    "No prediction (below 20%)",
                    (25, 88),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (180, 180, 180),
                    1,
                )

            cv2.putText(
                frame,
                "Press Q to quit",
                (15, frame.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

            cv2.imshow("SLTS - ASL Alphabet Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        detector.close()
        tts.close()

def detector_mode():
    import mediapipe as mp
    return mp.tasks.vision.RunningMode.VIDEO


if __name__ == "__main__":
    main()
