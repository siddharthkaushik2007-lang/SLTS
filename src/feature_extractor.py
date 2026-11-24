"""Convert MediaPipe's 21 hand landmarks into normalized numeric features."""

import numpy as np


def landmarks_to_features(hand_landmarks):
    """
    Input: 21 MediaPipe landmarks.
    Output: 63 normalized values (21 points x, y, z).

    Translation is removed by subtracting the wrist.
    Scale is normalized by the maximum distance from the wrist.
    """
    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
        dtype=np.float32
    )

    wrist = points[0].copy()
    points -= wrist

    scale = np.max(np.linalg.norm(points[:, :2], axis=1))
    if scale < 1e-6:
        return None

    points /= scale
    return points.flatten()


def draw_hand_landmarks(frame, hand_landmarks):
    """Draw only the detected hand skeleton on the preview frame."""
    import cv2

    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17)
    ]
    for a, b in connections:
        cv2.line(frame, points[a], points[b], (80, 220, 100), 2)
    for point in points:
        cv2.circle(frame, point, 4, (30, 120, 255), -1)
    return frame
