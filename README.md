# Gun Detection Flask App

This project is a Flask web app for firearm image detection. It supports:

- local detection with a trained Ultralytics YOLOv8 model
- optional hosted detection with a Roboflow workflow

## Dataset organization

Raw datasets are now organized under `dataset/raw/`:

```text
dataset/
└── raw/
    ├── kaggle_gun_detection/
    │   └── flat_yolo/
    ├── kaggle_weapon_detection/
    │   ├── metadata.csv
    │   ├── test/
    │   └── weapon_detection/
    └── roboflow_pistols/
        ├── README.dataset.txt
        ├── README.roboflow.txt
        └── coco_export/
```

## Supported training presets

### `pistol_detection`

Uses the sources you asked for:

- Kaggle Gun Detection Dataset
- Roboflow / Handgun-style pistols export already in your folder

This preset merges them into a single class:

- `pistol`

### `gun_types`

Uses the older Kaggle weapon detection dataset and keeps these classes:

- `automatic_rifle`
- `handgun`
- `shotgun`
- `smg`
- `sniper`

## Prepare a dataset

Default preset:

```bash
python scripts/prepare_dataset.py
```

Explicit presets:

```bash
python scripts/prepare_dataset.py --preset pistol_detection
python scripts/prepare_dataset.py --preset gun_types
```

This creates:

- `prepared_dataset/<preset>/`
- `dataset.<preset>.yaml`
- `dataset.yaml` pointing to the most recently prepared preset

## Train a local YOLO model

Pistol detection:

```bash
python scripts/train_model.py --preset pistol_detection --epochs 50 --imgsz 640 --batch 16
```

Gun types:

```bash
python scripts/train_model.py --preset gun_types --epochs 50 --imgsz 640 --batch 16
```

Outputs are written under:

```text
runs/detect/<preset>/
```

The Flask app automatically prefers:

1. `runs/detect/pistol_detection/weights/best.pt`
2. `runs/detect/gun_types/weights/best.pt`
3. `yolov8n.pt`

## Run the app with local YOLO

```bash
pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Run the app with Roboflow workflow

Roboflow documents workflow deployment with `InferenceHTTPClient` and `run_workflow()` in their official docs:

- https://docs.roboflow.com/workflows/deploy-a-workflow
- https://inference.roboflow.com/reference/inference_sdk/http/client/

Set these environment variables in PowerShell:

```powershell
$env:DETECTION_BACKEND="roboflow"
$env:ROBOFLOW_API_KEY="YOUR_API_KEY"
$env:ROBOFLOW_WORKSPACE="your-workspace"
$env:ROBOFLOW_WORKFLOW_ID="your-workflow-id"
$env:ROBOFLOW_CLASSES="pistol,pistol_c,pistol_nc"
python app.py
```

You can also use:

```powershell
$env:ROBOFLOW_CLASSES="hand,pistol,pistol_nc"
```

or:

```powershell
$env:ROBOFLOW_CLASSES="hand,pistol_c,pistol_nc"
```

The app will call your hosted workflow, draw the returned boxes on the uploaded image, and show the detected labels with confidence values.

## Notes

- I did not hardcode any Roboflow API key. You must provide your own credentials and workflow identifiers.
- The Roboflow workflow mode depends on the output shape of your workflow; this app assumes standard object-detection style predictions with box center coordinates and confidence values.
- The merged pistol preset improves local training quantity, but it is a single-class detector unless you add more reliably labeled subclasses.

## Sources

- Roboflow Workflows deployment docs: https://docs.roboflow.com/workflows/deploy-a-workflow
- Roboflow `run_workflow()` reference: https://inference.roboflow.com/reference/inference_sdk/http/client/
- Ultralytics docs: https://docs.ultralytics.com
