from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from features import sequence_to_features

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_FILE = MODEL_DIR / "sign_model.joblib"

X = []
y = []

class_dirs = sorted(
    [
        p for p in DATA_DIR.iterdir()
        if p.is_dir()
    ]
)

if not class_dirs:
    raise SystemExit(
        "No sign classes found. Run collect_sign.py first."
    )

print("Classes found:")

for class_dir in class_dirs:
    files = sorted(class_dir.glob("*.npy"))

    print(
        f"  {class_dir.name}: {len(files)} sequences"
    )

    for file in files:
        sequence = np.load(file)

        if sequence.ndim != 2:
            continue

        X.append(
            sequence_to_features(sequence)
        )

        y.append(class_dir.name)

if len(X) < 20:
    raise SystemExit(
        "Not enough training sequences."
    )

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)

if len(set(y)) < 2:
    raise SystemExit(
        "You need at least two different sign classes."
    )

# Stratification requires at least two samples per class.
class_counts = {
    label: int(np.sum(y == label))
    for label in set(y)
}

if min(class_counts.values()) < 2:
    raise SystemExit(
        "Every class needs at least 2 sequences."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print()
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

model = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions,
)

print()
print(f"Accuracy: {accuracy:.2%}")
print()
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
)

joblib.dump(
    {
        "model": model,
        "sequence_length": 30,
    },
    MODEL_FILE,
)

print()
print(f"Saved model to: {MODEL_FILE}")
