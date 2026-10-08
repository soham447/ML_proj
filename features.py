import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands


def create_detector():
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
    )


def extract_frame_features(result):
    """Create a fixed feature vector for one frame.

    Each hand has 21 landmarks * XYZ = 63 values.
    Two hands = 126 values.

    The handedness output is used to put left/right hands into
    consistent slots when available.
    """

    features = np.zeros(126, dtype=np.float32)

    if not result.multi_hand_landmarks:
        return features

    hands_data = []

    for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
        points = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32,
        )

        # Wrist becomes origin.
        points -= points[0]

        # Scale normalization.
        scale = np.max(np.linalg.norm(points[:, :2], axis=1))

        if scale > 1e-6:
            points /= scale

        label = None

        if result.multi_handedness and i < len(result.multi_handedness):
            label = result.multi_handedness[i].classification[0].label

        hands_data.append((label, points.flatten()))

    # Assign slots consistently.
    left = None
    right = None

    for label, data in hands_data:
        if label == "Left":
            left = data
        elif label == "Right":
            right = data

    # If handedness is unavailable, preserve detection order.
    if left is None and right is None:
        if len(hands_data) >= 1:
            left = hands_data[0][1]

        if len(hands_data) >= 2:
            right = hands_data[1][1]

    if left is not None:
        features[:63] = left

    if right is not None:
        features[63:126] = right

    return features


def process_frame(detector, frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)
    return result


def sequence_to_features(sequence):
    """Convert a sequence into fixed-size features.

    Each frame has 126 landmark values.

    We use:
      - normalized landmark values
      - first-order motion
      - mean absolute motion
      - standard deviation

    This lets a classical classifier see both pose and movement.
    """

    x = np.asarray(sequence, dtype=np.float32)

    if len(x) == 0:
        return np.zeros(504, dtype=np.float32)

    # Fixed temporal length is guaranteed by collector/recognizer.
    position = x

    velocity = np.diff(
        x,
        axis=0,
        prepend=x[:1],
    )

    mean_motion = np.mean(
        np.abs(velocity),
        axis=0,
    )

    std_motion = np.std(
        velocity,
        axis=0,
    )

    return np.concatenate(
        [
            position.flatten(),
            velocity.flatten(),
            mean_motion,
            std_motion,
        ]
    ).astype(np.float32)
