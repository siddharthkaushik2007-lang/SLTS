# SLTS — ASL Letter and Digit Recognition (A–Z, 0–9)

A beginner-friendly, one-hand webcam prototype that recognizes ASL alphabet
and digit hand configurations using MediaPipe landmarks and a Random Forest
classifier.

## What it does

1. Downloads the ASL Alphabet image dataset from Kaggle using KaggleHub and the
   sign-language digits dataset from GitHub.
2. Detects one hand in each dataset image using MediaPipe.
3. Converts 21 hand landmarks into 63 normalized numeric features.
4. Trains a Random Forest model for A–Z and 0–9.
5. Opens the webcam and displays the predicted letter.

The webcam pipeline uses a hand-landmark detector configured for one hand.
It does not perform face recognition or body-pose recognition. Frames are
processed locally and are not saved by this application.

## Important ASL limitation: J and Z

In ASL, J and Z are dynamic letters: the signer traces a movement. The selected
Kaggle dataset consists of still images, so it cannot teach the model the full
movement trajectory. This starter implementation classifies the static images
labelled J and Z as still poses; it is NOT a validated dynamic J/Z recognizer.
A true 26-letter ASL system needs labelled video/sequence examples for J and Z
and a temporal model (for example, an LSTM or sequence classifier).

## Requirements

- Windows 10/11, Python 3.10 or 3.11 recommended
- Webcam
- Internet connection for initial dataset and MediaPipe model downloads
- Kaggle account/API access

## Setup (Windows PowerShell)

Open PowerShell in the SLTS folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Kaggle API access

1. Sign in to Kaggle.
2. Open your Kaggle account settings and create an API token.
3. Configure the token as instructed by Kaggle/KaggleHub.
4. Do not share `kaggle.json` or API credentials.

KaggleHub will download `grassknoted/asl-alphabet` to its local cache. The
digit dataset is downloaded from the
[Arda Mavi Sign Language Digits Dataset](https://github.com/ardamavi/Sign-Language-Digits-Dataset)
into `datasets/raw`.

## Run the project

Run each command from the project root, with the virtual environment active:

```powershell
python -m src.dataset_downloader
python -m src.dataset_processor
python -m src.train_model
python -m src.evaluate_model
python main.py
```

The alphabet dataset download is large (approximately 1.1 GB). Landmark
extraction can take a while. `MAX_IMAGES_PER_CLASS` in `config.py` is set to
1500 to make the first run manageable. Increase it or set it to `None` for
more training images, subject to available disk space and time.

## Project files

- `config.py`: paths, thresholds and settings
- `src/dataset_downloader.py`: Kaggle dataset download
- `src/hand_detector.py`: MediaPipe hand detector
- `src/feature_extractor.py`: normalized 63-value feature vector
- `src/dataset_processor.py`: image-to-landmark CSV pipeline
- `src/train_model.py`: Random Forest training and evaluation
- `src/evaluate_model.py`: metrics and confusion matrix
- `src/webcam_recognition.py`: live webcam prediction
- `main.py`: application entry point

## Troubleshooting

- `Could not open webcam`: close other camera applications and check Windows
  camera permissions. Try changing `cv2.VideoCapture(0)` to `1`.
- Missing class folders: run the downloader and verify the alphabet dataset
  has `A` through `Z` folders and the digit dataset has `0` through `9` folders.
- No hand landmarks extracted: use clear, front-facing hand images. The dataset
  may include images that the hand detector cannot process.
- Low live accuracy: improve lighting, keep the hand fully visible, use a plain
  background, and collect additional representative training examples.
- Package installation issues: use Python 3.10/3.11 in a fresh virtual
  environment and install the listed requirements.

## Evaluation note

The test split in this starter version is a random image-level split. Images
from the same source collection may be visually similar across train and test,
so this accuracy is not a reliable measure of performance on unseen people.
For a stronger evaluation, use a separate signer/session-based test set.

## Dataset attribution

ASL Alphabet dataset by grassknoted:
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

Sign Language Digits Dataset by Arda Mavi and students at Turkey Ankara
Ayrancı Anadolu High School:
https://github.com/ardamavi/Sign-Language-Digits-Dataset

The digit dataset asks users of the data in research projects to cite:
Mavi, A. (2020), “A New Dataset and Proposed Convolutional Neural Network
Architecture for Classification of American Sign Language Digits,”
arXiv:2011.08927.

Review each dataset's current license, terms, and attribution requirements
before redistributing the data or project.
