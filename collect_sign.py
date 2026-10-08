import json
import time
from pathlib import Path

import cv2
import numpy as np

from features import create_detector, process_frame, extract_frame_features

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SEQUENCE_LENGTH = 30
SEQUENCES_PER_RUN = 50

print("=" * 60)
print("CUSTOM GENERAL SIGN DATA COLLECTOR")
print("=" * 60)
print()
print("You can enter ANY name:")
print("hello, thank_you, yes, no, stop, help, etc.")
print()

label = input("Sign name: ").strip().lower()

if not label:
    raise SystemExit("Sign name cannot be empty.")

safe_label = "".join(
    c if c.isalnum() or c in "_-" else "_"
    for c in label
)

label_dir = DATA_DIR / safe_label
label_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise SystemExit("Could not open webcam.")

detector = create_detector()

print()
print(f"Recording class: {safe_label}")
print(f"Each sequence contains {SEQUENCE_LENGTH} frames.")
print(f"Target sequences: {SEQUENCES_PER_RUN}")
print()
print("Press R to record.")
print("Press Q to quit.")

sequence_number = len(list(label_dir.glob("*.npy")))

while True:
    ok, frame = cap.read()

    if not ok:
        break

    frame = cv2.flip(frame, 1)

    cv2.putText(
        frame,
        f"Sign: {safe_label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"Sequences: {sequence_number}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        "R = record sequence | Q = quit",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Custom Sign Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("r"):
        sequence = []

        print(
            f"Recording sequence {sequence_number + 1}..."
        )

        for i in range(SEQUENCE_LENGTH):
            ok, frame = cap.read()

            if not ok:
                break

            frame = cv2.flip(frame, 1)

            result = process_frame(detector, frame)
            features = extract_frame_features(result)

            sequence.append(features)

            progress = int(
                300 * (i + 1) / SEQUENCE_LENGTH
            )

            cv2.rectangle(
                frame,
                (20, 140),
                (320, 165),
                (80, 80, 80),
                -1,
            )

            cv2.rectangle(
                frame,
                (20, 140),
                (20 + progress, 165),
                (0, 255, 0),
                -1,
            )

            cv2.putText(
                frame,
                "RECORDING",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

            cv2.imshow(
                "Custom Sign Collector",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if len(sequence) == SEQUENCE_LENGTH:
            filename = label_dir / f"{sequence_number:05d}.npy"

            np.save(
                filename,
                np.asarray(sequence, dtype=np.float32),
            )

            sequence_number += 1

            print(f"Saved {filename}")

        # Small pause between samples.
        time.sleep(0.2)

cap.release()
cv2.destroyAllWindows()
detector.close()

print("Collection finished.")
