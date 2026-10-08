# Custom General Sign Recognition System

A webcam-based custom gesture/sign recognizer.

The model learns short sequences of hand landmarks, so it can recognize both static and moving gestures better than a single-frame classifier.

## How it works

```text
Webcam
  |
  v
MediaPipe Hands
  |
  v
21 landmarks x up to 2 hands
  |
  v
Normalized motion sequence
  |
  v
Random Forest sequence classifier
  |
  v
Sign name
  |
  v
Sentence builder
```

The feature vector contains both hands and motion information across multiple frames.

Install:

```bash
pip install -r requirements.txt
```

## Step 1: Record signs

Run:

```bash
python collect_sign.py
```

Example:

```text
Sign name: hello
```

The webcam opens.

Press `R` to record a sequence.

Perform the sign naturally for about 1-2 seconds.

The program automatically saves one sequence.

Record at least 20 sequences for each sign.

You can create as many custom classes as you want.

## Step 2: Train

Run:

```bash
python train_model.py
```

The program automatically discovers every sign in the dataset.

It saves:

```text
models/sign_model.joblib
```

## Step 3: Recognize

Run:

```bash
python recognize.py
```

Perform a learned sign.

The recognized sign is added to the text after a stable prediction.

Example:

```text
HELLO THANK_YOU HELP
```

## Controls

During recognition:

- `SPACE` -> add a space
- `BACKSPACE` -> remove the last word/sign
- `C` -> clear sentence
- `Q` -> quit

During recording:

- `R` -> record one sequence
- `Q` -> quit
