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
| `backend/` | FastAPI source |
| `frontend/` | Streamlit source |
| `face_mask_model.keras` | Complete classifier loaded by the backend |
| `resnet50_local.keras` | Optional base-model export referenced in the notebook |
| `main_resnet.ipynb` | ResNet50 training outline and local inference experiments |
| `main.ipynb` | Additional notebook; not reviewed for this documentation |
| `data/` | Local dataset |
| `facepred.png` | Image asset visible in the project structure |
| `archive.zip` | Dataset archive; excluded from Git |

The commands below assume you save the supplied backend as `backend/main.py` and frontend as `frontend/app.py`. If your filenames differ, adjust the commands accordingly.

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

Keep `face_mask_model.keras` in the project root. The backend loads this relative path, so start both commands from the main `FaceMaskDetection` folder.

Terminal 1, with the environment active:

```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2, with the same environment active:

```bash
python -m streamlit run frontend/app.py
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

1. Obtain the dataset used for the project and place images in `data/with_mask/` and `data/without_mask/`. Its download source and licence have not been supplied.
2. Open `main_resnet.ipynb` in VS Code or run `python -m notebook`.
3. Uncomment the dataset-loading, model-building, preprocessing-map, compilation, training, and model-export cells, including `from tensorflow.keras import models, layers`.
4. Run those cells in order. Verify `train_ds.class_names` before mapping preprocessing, so the class order agrees with the API labels.
5. Save the resulting classifier as `face_mask_model.keras` in the project root.

The optional `resnet50_local.keras` export is not loaded by the supplied backend. The complete classifier is the artifact needed for inference.

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

## GitHub preparation

The accompanying `.gitignore` excludes `archive.zip`, secrets, virtual environments, and caches. It leaves the model files and dataset eligible for upload pending file-size checks. Check individual model sizes and dataset size before committing. Include dataset attribution and only redistribute images you have permission to share.
