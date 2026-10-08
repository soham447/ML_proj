import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np

from features import (
    create_detector,
    process_frame,
    extract_frame_features,
    sequence_to_features,
)

MODEL_FILE = Path("models/sign_model.joblib")

if not MODEL_FILE.exists():
    raise SystemExit(
        "Model not found. Run train_model.py first."
    )

saved = joblib.load(MODEL_FILE)

model = saved["model"]
SEQUENCE_LENGTH = saved["sequence_length"]

detector = create_detector()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise SystemExit("Could not open webcam.")

# Current rolling sequence.
sequence = deque(
    maxlen=SEQUENCE_LENGTH
)

# Prediction smoothing.
prediction_history = deque(
    maxlen=8
)

last_sign = None
last_sign_time = 0

# Time before a recognized sign is added to text.
SIGN_COOLDOWN = 1.2

# Minimum probability.
CONFIDENCE_THRESHOLD = 0.60

sentence = ""


def add_sign(sign):
    global sentence

    if sentence:
        sentence += " "

    sentence += sign


def delete_last():
    global sentence

    sentence = sentence.rstrip()

    if sentence:
        sentence = sentence.rsplit(" ", 1)[0]


def clear_sentence():
    global sentence
    sentence = ""


print("=" * 60)
print("CUSTOM GENERAL SIGN RECOGNITION")
print("=" * 60)
print()
print("Perform a learned sign in front of the camera.")
print()
print("Controls:")
print("SPACE      -> add a space")
print("BACKSPACE  -> delete previous sign/word")
print("C          -> clear")
print("Q          -> quit")


while True:
    ok, frame = cap.read()

    if not ok:
        break

    frame = cv2.flip(frame, 1)

    result = process_frame(
        detector,
        frame,
    )

    features = extract_frame_features(
        result
    )

    sequence.append(features)

    prediction = "Waiting..."
    confidence = 0.0

    if len(sequence) == SEQUENCE_LENGTH:
        model_features = sequence_to_features(
            list(sequence)
        ).reshape(1, -1)

        probabilities = model.predict_proba(
            model_features
        )[0]

        best = int(
            np.argmax(probabilities)
        )

        prediction = str(
            model.classes_[best]
        )

        confidence = float(
            probabilities[best]
        )

        if confidence >= CONFIDENCE_THRESHOLD:
            prediction_history.append(
                prediction
            )

            # Majority vote over recent predictions.
            values, counts = np.unique(
                prediction_history,
                return_counts=True,
            )

            prediction = str(
                values[np.argmax(counts)]
            )

            stable_count = max(counts)

            now = time.monotonic()

            # Require repeated agreement before insertion.
            if (
                stable_count >= 6
                and (
                    prediction != last_sign
                    or now - last_sign_time
                    > SIGN_COOLDOWN
                )
            ):
                add_sign(prediction)

                last_sign = prediction
                last_sign_time = now

                prediction_history.clear()

    # Draw hand landmarks.
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            import mediapipe as mp

            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand,
                mp.solutions.hands.HAND_CONNECTIONS,
            )

    h, w = frame.shape[:2]

    # Text panel.
    cv2.rectangle(
        frame,
        (10, 10),
        (w - 10, 130),
        (0, 0, 0),
        -1,
    )

    display_sentence = sentence

    if len(display_sentence) > 55:
        display_sentence = (
            "..." + display_sentence[-52:]
        )

    cv2.putText(
        frame,
        "TEXT: " + display_sentence,
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"Sign: {prediction}",
        (25, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.0%}",
        (25, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    # Instructions.
    cv2.putText(
        frame,
        "SPACE=space  BACKSPACE=delete  C=clear  Q=quit",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    cv2.imshow(
        "Custom General Sign Recognition",
        frame,
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("c"):
        clear_sentence()

    elif key == 32:
        if sentence and not sentence.endswith(" "):
            sentence += " "

    elif key == 8:
        delete_last()


cap.release()
cv2.destroyAllWindows()
detector.close()

print()
print("Recognized text:")
print(sentence)
