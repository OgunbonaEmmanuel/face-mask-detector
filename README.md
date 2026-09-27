# Face Mask Detector

A computer vision application with a FastAPI backend and Streamlit interface for classifying images as **MASK ON** or **MASK OFF**. It supports uploaded images and live frames from a local webcam.

The project includes a ResNet50 transfer-learning notebook and uses an OpenCV Haar cascade to locate a face in live frames.

## Features

- Upload JPG, JPEG, or PNG images for classification.
- Start a local webcam using **GO LIVE**.
- Detect and classify the first face returned by the Haar cascade in each live frame.
- Display green boxes for MASK ON and red boxes for MASK OFF.
- Serve predictions through two FastAPI endpoints.

## Project layout

Place this README, requirements, and `.gitignore` in the main `FaceMaskDetection` folder.

| Path | Purpose |
| --- | --- |
| `backend/main.py` | FastAPI prediction service |
| `frontend/main.py` | Streamlit interface |
| `face_mask_model.keras` | Classifier used when starting the backend from the project root |
| `backend/face_mask_model.keras` | Additional classifier copy included in the repository |
| `resnet50_local.keras` | Optional local base-model export; excluded from GitHub |
| `main_resnet.ipynb` | ResNet50 training outline and local inference experiments |
| `main.ipynb` | Additional notebook; not reviewed for this documentation |
| `data/` | Local training images; excluded from GitHub |
| `facepred.png` | Image asset visible in the project structure |
| `archive.zip` | Dataset archive; excluded from Git |

The training dataset, `archive.zip`, and `resnet50_local.keras` are not included in the repository. They are not needed to run predictions with the complete classifier. Retraining requires obtaining the dataset separately. Python caches and local environment files are also excluded.

## Installation

From the project root:

```bash
python -m venv .venv
```

Activate the environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the packages:

```bash
python -m pip install -r requirements.txt
```

Requirements cover the supplied application and `main_resnet.ipynb`. OpenCV's desktop package is included because the notebook uses `cv2.imshow()`. Versions are unpinned because a tested environment was not supplied. Use a Python version supported by your TensorFlow release and an environment compatible with the saved model.

## Run the application

The repository includes classifier copies in the project root and in `backend/`. The supplied backend loads `face_mask_model.keras` relative to its working directory. The commands below use the root copy: start both terminals in the main `FaceMaskDetection` folder. Only one matching classifier file is needed for a chosen startup directory.

Terminal 1, with the environment active:

```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2, with the same environment active:

```bash
python -m streamlit run frontend/main.py
```

Open the Streamlit address shown in the terminal. Upload an image or click **GO LIVE**. The frontend contacts `http://127.0.0.1:8000`.

The webcam is opened through `cv2.VideoCapture(0)` on the computer running Streamlit. It does not capture a remote visitor's browser webcam. No API key is required.

The supplied live loop has no Stop button. For local testing, interrupt the Streamlit process from its terminal when finished. A Stop control and `try/finally` camera cleanup are recommended improvements.

## API behavior

Both routes accept a multipart upload with the field name `file`.

| Endpoint | Behavior |
| --- | --- |
| `POST /image_predict` | Classifies the entire uploaded image without detecting or cropping faces |
| `POST /vid_predict` | Detects faces in a frame and classifies only the first detected face |

For classification, the backend converts BGR to RGB, resizes to 224 × 224, casts to float32, applies ResNet50 `preprocess_input`, and adds a batch dimension.

Scores below 0.5 return MASK ON. Scores greater than or equal to 0.5 return MASK OFF.

Example successful live response:

```json
{
  "prediction": "MASK OFF",
  "confidence": 0.91,
  "bbox": [100, 60, 120, 120]
}
```

The bounding box is `[x, y, width, height]`. The `confidence` field always contains the raw MASK OFF sigmoid score, even when the label is MASK ON. It is not currently a label-specific confidence measure. For MASK ON, the complementary class score is `1 - confidence`; calibration has not been evaluated.

The video route returns `Invalid image` or `No face detected`, with a score of zero and no bounding box, when appropriate. These are status messages, not class predictions.

## Model notebook

The supplied `main_resnet.ipynb` contains this training configuration as commented-out code:

| Setting | Configuration |
| --- | --- |
| Dataset | Images loaded from `data/` |
| Split | 80% training, 20% validation, seed 123 |
| Input / batch | 224 × 224 RGB / 32 images |
| Backbone | ImageNet-pretrained ResNet50, `include_top=False`, frozen |
| Classifier head | Global average pooling, Dense 128 ReLU, Dropout 0.5, Dense 1 sigmoid |
| Optimization | Adam and binary cross-entropy |
| Training duration | Five epochs specified |

The notebook labels class 0 as MASK ON and class 1 as MASK OFF, and references `data/with_mask/` and `data/without_mask/`.

The saved notebook does not contain training metrics, so no accuracy claim is made. Its active cells load an existing classifier and run image/webcam experiments. The saved model binary was not inspected to independently confirm the architecture.

### Reproduce training

1. Obtain the original training dataset separately; the training images are not included in this repository. Place images in `data/with_mask/` and `data/without_mask/`. An exact download source and licence have not yet been documented, so exact dataset reproduction is not currently provided.
2. Open `main_resnet.ipynb` in VS Code or run `python -m notebook`.
3. Uncomment the dataset-loading, model-building, preprocessing-map, compilation, training, and model-export cells, including `from tensorflow.keras import models, layers`.
4. Run those cells in order. Verify `train_ds.class_names` before mapping preprocessing, so the class order agrees with the API labels.
5. Save the resulting classifier as `face_mask_model.keras` in the project root.

The optional `resnet50_local.keras` export is excluded from GitHub and is not loaded by the backend. To rebuild from the commented training code, create the ImageNet-pretrained backbone as shown in the notebook; an existing local base-model export is not required. The complete `face_mask_model.keras` classifier is sufficient for inference.

## Known limitations

- Live inference handles one detected face per frame, not all faces.
- Haar detection may miss obscured or angled faces, including faces wearing masks.
- Requests and inference run sequentially for each frame; latency limits the live frame rate.
- Network calls have no timeout, and exceptions can bypass `cap.release()`.
- The image endpoint does not validate failed image decoding before color conversion.
- The frontend does not handle all connection and response-parsing errors.
- The notebook's image helper passes OpenCV BGR input to preprocessing without the RGB conversion used by the API. Align this before comparing results.
- The notebook webcam experiment may use an undefined or stale `y_pred` when no face is detected.

This is a portfolio prototype. The source and notebook were reviewed, but the model, API, and webcam pipeline were not executed during documentation preparation.

## Files excluded from GitHub

The repository excludes:

- `data/`: training images retained locally.
- `archive.zip`: the dataset archive.
- `resnet50_local.keras`: optional base-model export.
- Python caches, virtual environments, and secret environment files.

The application source, notebooks, documentation, requirements, and complete classifier remain included. Ignoring local training files does not prevent inference with the saved classifier. Anyone wishing to retrain must obtain the dataset separately and follow the notebook setup above.
