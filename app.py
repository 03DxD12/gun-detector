import os
from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, request, redirect, url_for, session, make_response, jsonify
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MODEL_CANDIDATES = [
    BASE_DIR / "runs" / "detect" / "weapon_accuracy_fix" / "weights" / "best.pt",
    BASE_DIR / "runs" / "detect" / "gun_types_v3" / "weights" / "best.pt",
    BASE_DIR / "runs" / "detect" / "gun_types_v2" / "weights" / "best.pt",
    BASE_DIR / "runs" / "detect" / "gun_types" / "weights" / "best.pt",
    BASE_DIR / "runs" / "detect" / "weapon_detector_clean" / "weights" / "best.pt",
    BASE_DIR / "runs" / "detect" / "weapon_detector" / "weights" / "best.pt",
]
RUNS_DIR = BASE_DIR / "runs" / "detect"
PISTOL_ASSIST_PATH = BASE_DIR / "runs" / "detect" / "pistol_detection" / "weights" / "best.pt"

app = Flask(__name__)
app.secret_key = "weapon-detection-secret"
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DETECTION_BACKEND = os.getenv("DETECTION_BACKEND", "yolo").lower()
ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "")
ROBOFLOW_WORKFLOW_ID = os.getenv("ROBOFLOW_WORKFLOW_ID", "")
ROBOFLOW_CLASSES = os.getenv("ROBOFLOW_CLASSES", "pistol")
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.85"))
SINGLE_WEAPON_MODE = True  # Focus on exactly one weapon for maximum precision
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "1280"))
YOLO_AUGMENT = os.getenv("YOLO_AUGMENT", "true").lower() == "true"
YOLO_TILE_SIZE = int(os.getenv("YOLO_TILE_SIZE", "960"))
YOLO_TILE_OVERLAP = float(os.getenv("YOLO_TILE_OVERLAP", "0.25"))
YOLO_REFINE_PADDING = float(os.getenv("YOLO_REFINE_PADDING", "0.18"))
LABEL_ALIASES = {
    "handgun": "PISTOL",
    "pistol": "PISTOL",
    "glock": "PISTOL",
    "automatic_rifle": "AR",
    "ar": "AR",
    "m4": "AR",
    "ak47": "AR",
    "shotgun": "RIFLE",
    "rifle": "RIFLE",
    "smg": "SMG",
    "mp5": "SMG",
    "p90": "SMG",
    "sniper": "SNIPER",
    "sniper_rifle": "SNIPER",
    "knife": "KNIFE",
    "sword": "KNIFE",
    "bazooka": "HEAVY_WEAPON",
    "grenade_launcher": "HEAVY_WEAPON",
    "weapon": "WEAPON",
    "gun": "WEAPON",
}


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def resolve_model_path() -> Path:
    explicit_path = os.getenv("MODEL_PATH")
    if explicit_path:
        candidate = Path(explicit_path)
        if candidate.exists():
            return candidate

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate

    discovered = [
        path
        for path in RUNS_DIR.rglob("best.pt")
        if path.is_file() and "weights" in path.parts
    ]
    if discovered:
        discovered.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return discovered[0]

    raise FileNotFoundError(
        "No trained weapon model found. Train one first or set MODEL_PATH to a custom best.pt file."
    )


MODEL_LOAD_ERROR = None
MODEL_PATH = None
YOLO_MODEL = None
PISTOL_ASSIST_MODEL = None

if DETECTION_BACKEND == "yolo":
    try:
        MODEL_PATH = resolve_model_path()
        YOLO_MODEL = YOLO(str(MODEL_PATH))
        if PISTOL_ASSIST_PATH.exists():
            PISTOL_ASSIST_MODEL = YOLO(str(PISTOL_ASSIST_PATH))
    except Exception as exc:
        MODEL_LOAD_ERROR = str(exc)


def get_backend_label() -> str:
    if DETECTION_BACKEND == "roboflow":
        return f"roboflow:{ROBOFLOW_WORKFLOW_ID or 'unconfigured'}"
    if MODEL_PATH is not None:
        return f"yolo:{MODEL_PATH.name}"
    return "yolo:unconfigured"


def get_supported_classes():
    if DETECTION_BACKEND == "roboflow":
        return [item.strip() for item in ROBOFLOW_CLASSES.split(",") if item.strip()]

    if YOLO_MODEL is None:
        return []

    names = YOLO_MODEL.names
    if isinstance(names, dict):
        classes = [names[idx] for idx in sorted(names)]
    else:
        classes = list(names)
    # Deduplicate normalized labels
    return sorted(list(set(normalize_label(name) for name in classes)))


def normalize_label(label: str) -> str:
    return LABEL_ALIASES.get(label, label)


def box_area(prediction):
    return prediction["width"] * prediction["height"]


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a["bbox"]
    bx1, by1, bx2, by2 = box_b["bbox"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def clamp(value, low, high):
    return max(low, min(value, high))


def shift_prediction(prediction, offset_x=0.0, offset_y=0.0):
    shifted = dict(prediction)
    shifted["x"] = prediction["x"] + offset_x
    shifted["y"] = prediction["y"] + offset_y
    shifted["bbox"] = [
        prediction["bbox"][0] + offset_x,
        prediction["bbox"][1] + offset_y,
        prediction["bbox"][2] + offset_x,
        prediction["bbox"][3] + offset_y,
    ]
    return shifted


def run_model(model, image_source, conf, imgsz, augment):
    result = model(
        image_source,
        conf=conf,
        imgsz=imgsz,
        augment=augment,
        verbose=False,
    )[0]
    predictions = extract_yolo_predictions(result, min_conf=conf)
    image_height, image_width = result.orig_shape
    return predictions, image_width, image_height


def generate_tiles(image_width, image_height, tile_size, overlap):
    if image_width <= tile_size and image_height <= tile_size:
        return [(0, 0, image_width, image_height)]

    stride = max(1, int(tile_size * (1 - overlap)))
    tiles = []
    y = 0
    while True:
        x = 0
        bottom = min(image_height, y + tile_size)
        if bottom - y < tile_size and bottom == image_height:
            y = max(0, image_height - tile_size)
            bottom = image_height
        while True:
            right = min(image_width, x + tile_size)
            if right - x < tile_size and right == image_width:
                x = max(0, image_width - tile_size)
                right = image_width
            tiles.append((x, y, right, bottom))
            if right >= image_width:
                break
            x += stride
        if bottom >= image_height:
            break
        y += stride

    deduped = []
    seen = set()
    for tile in tiles:
        if tile not in seen:
            seen.add(tile)
            deduped.append(tile)
    return deduped


def merge_predictions(predictions, iou_threshold=0.45):
    ordered = sorted(predictions, key=lambda item: item["confidence"], reverse=True)
    merged = []
    for prediction in ordered:
        skip = False
        for chosen in merged:
            # Cross-class suppression: if two weapon boxes overlap heavily,
            # keep only the one with higher confidence regardless of class.
            if compute_iou(prediction, chosen) >= iou_threshold:
                skip = True
                break
        if not skip:
            merged.append(prediction)
    return merged


def filter_weapon_predictions(predictions, image_width, image_height):
    image_area = image_width * image_height
    filtered = []

    class_area_limits = {
        "PISTOL": 0.85,
        "SMG": 0.85,
        "AR": 0.85,
        "RIFLE": 0.85,
        "SNIPER": 0.85,
        "KNIFE": 0.85,
        "HEAVY_WEAPON": 0.85,
    }

    for prediction in predictions:
        area_ratio = box_area(prediction) / image_area if image_area else 0
        width_ratio = prediction["width"] / image_width if image_width else 0
        height_ratio = prediction["height"] / image_height if image_height else 0

        max_area_ratio = class_area_limits.get(prediction["class"], 0.35)
        if area_ratio > max_area_ratio:
            continue
        if width_ratio > 0.85 and height_ratio > 0.85:
            continue
        if prediction["class"] == "handgun" and width_ratio > 0.45:
            continue
        if prediction["class"] == "handgun" and height_ratio > 0.6:
            continue

        filtered.append(prediction)

    filtered.sort(key=lambda item: item["confidence"], reverse=True)
    kept = []
    for prediction in filtered:
        reject = False
        for chosen in kept:
            iou = compute_iou(prediction, chosen)
            if prediction["class"] == chosen["class"]:
                if iou > 0.35:
                    reject = True
                    break
                if iou > 0.10 and box_area(prediction) > box_area(chosen) * 1.8:
                    reject = True
                    break

            # If a lower-confidence box of another class overlaps heavily, prefer the stronger one.
            if iou > 0.55 and prediction["confidence"] + 0.1 < chosen["confidence"]:
                reject = True
                break
        if not reject:
            kept.append(prediction)

    return kept


def tile_detect_with_yolo(image_path: Path):
    image = Image.open(image_path)
    image_width, image_height = image.size
    predictions = []

    full_predictions, _, _ = run_model(
        YOLO_MODEL,
        str(image_path),
        conf=YOLO_CONF,
        imgsz=YOLO_IMGSZ,
        augment=YOLO_AUGMENT,
    )
    predictions.extend(full_predictions)

    for left, top, right, bottom in generate_tiles(
        image_width,
        image_height,
        tile_size=YOLO_TILE_SIZE,
        overlap=YOLO_TILE_OVERLAP,
    ):
        tile = image.crop((left, top, right, bottom))
        tile_predictions, _, _ = run_model(
            YOLO_MODEL,
            tile,
            conf=max(0.05, YOLO_CONF * 0.75),
            imgsz=YOLO_TILE_SIZE,
            augment=YOLO_AUGMENT,
        )
        for prediction in tile_predictions:
            predictions.append(shift_prediction(prediction, left, top))

    return image_width, image_height, merge_predictions(predictions)


def crop_box(prediction, image_width, image_height, padding_ratio=YOLO_REFINE_PADDING):
    left, top, right, bottom = prediction["bbox"]
    pad_x = prediction["width"] * padding_ratio
    pad_y = prediction["height"] * padding_ratio
    return (
        int(clamp(left - pad_x, 0, image_width)),
        int(clamp(top - pad_y, 0, image_height)),
        int(clamp(right + pad_x, 0, image_width)),
        int(clamp(bottom + pad_y, 0, image_height)),
    )


def refine_prediction_classes(image_path: Path, predictions, image_width, image_height):
    if not predictions:
        return predictions

    image = Image.open(image_path).convert("RGB")
    refined = []
    for prediction in predictions:
        crop = crop_box(prediction, image_width, image_height)
        left, top, right, bottom = crop
        if right - left < 24 or bottom - top < 24:
            refined.append(prediction)
            continue

        crop_image = image.crop(crop)
        crop_predictions, _, _ = run_model(
            YOLO_MODEL,
            crop_image,
            conf=max(0.05, YOLO_CONF * 0.5),
            imgsz=max(YOLO_IMGSZ, 1280),
            augment=False,
        )

        best_local = None
        for local_prediction in crop_predictions:
            local_area = box_area(local_prediction)
            crop_area = max(1.0, (right - left) * (bottom - top))
            if local_area / crop_area > 0.9:
                continue
            if best_local is None or local_prediction["confidence"] > best_local["confidence"]:
                best_local = local_prediction

        updated = dict(prediction)
        if best_local is not None:
            updated["class"] = best_local["class"]
            updated["confidence"] = max(prediction["confidence"], best_local["confidence"])
            local_left, local_top, local_right, local_bottom = best_local["bbox"]
            updated["bbox"] = [
                local_left + left,
                local_top + top,
                local_right + left,
                local_bottom + top,
            ]
            updated["x"] = (updated["bbox"][0] + updated["bbox"][2]) / 2
            updated["y"] = (updated["bbox"][1] + updated["bbox"][3]) / 2
            updated["width"] = updated["bbox"][2] - updated["bbox"][0]
            updated["height"] = updated["bbox"][3] - updated["bbox"][1]
        refined.append(updated)

    return refined


def assist_handgun_predictions(image_path: Path, predictions):
    if PISTOL_ASSIST_MODEL is None:
        return predictions

    long_gun_predictions = {
        "AR",
        "RIFLE",
        "SMG",
        "SNIPER",
        "HEAVY_WEAPON",
    }
    if any(
        prediction["class"] in long_gun_predictions and prediction["confidence"] >= 0.3
        for prediction in predictions
    ):
        return predictions
    if any(
        prediction["class"] == "PISTOL" and prediction["confidence"] >= 0.6
        for prediction in predictions
    ):
        return predictions

    assist_predictions, _, _ = run_model(
        PISTOL_ASSIST_MODEL,
        str(image_path),
        conf=max(0.12, YOLO_CONF),
        imgsz=max(960, YOLO_IMGSZ),
        augment=False,
    )
    for assist_prediction in assist_predictions:
        if assist_prediction["width"] <= 0 or assist_prediction["height"] <= 0:
            continue
        assist_prediction["class"] = "handgun"
    assist_predictions = [
        prediction
        for prediction in assist_predictions
        if prediction["confidence"] >= 0.35
    ]
    return merge_predictions(predictions + assist_predictions, iou_threshold=0.4)


def extract_yolo_predictions(result, min_conf=0.4, allowed_classes=None):
    predictions = []
    names = result.names
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            conf = float(box.conf[0].item())
            if conf < min_conf:
                continue

            cls_id = int(box.cls[0].item())
            cls_name = names.get(cls_id, str(cls_id))
            
            if allowed_classes and cls_name not in allowed_classes:
                continue

            predictions.append({
                "class": normalize_label(cls_name),
                "confidence": round(conf, 2),
                "x": float(box.xywh[0][0].item()),
                "y": float(box.xywh[0][1].item()),
                "width": float(box.xywh[0][2].item()),
                "height": float(box.xywh[0][3].item()),
                "bbox": [
                    float(box.xyxy[0][0].item()),
                    float(box.xyxy[0][1].item()),
                    float(box.xyxy[0][2].item()),
                    float(box.xyxy[0][3].item())
                ]
            })
    return predictions


def detect_with_yolo(output_path: Path):
    if YOLO_MODEL is None:
        raise RuntimeError(MODEL_LOAD_ERROR or "YOLO model is not loaded.")

    image_width, image_height, predictions = tile_detect_with_yolo(output_path)
    predictions = refine_prediction_classes(output_path, predictions, image_width, image_height)
    predictions = assist_handgun_predictions(output_path, predictions)
    predictions = merge_predictions(predictions, iou_threshold=0.35)
    predictions = filter_weapon_predictions(predictions, image_width, image_height)
    draw_predictions(output_path, predictions)
    return predictions


def extract_predictions(payload):
    predictions = []
    if isinstance(payload, dict):
        value = payload.get("predictions")
        if isinstance(value, list):
            predictions.extend(value)
        for item in payload.values():
            predictions.extend(extract_predictions(item))
    elif isinstance(payload, list):
        for item in payload:
            predictions.extend(extract_predictions(item))
    return predictions


def draw_predictions(output_path: Path, predictions):
    image = Image.open(output_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for prediction in predictions:
        x = float(prediction.get("x", 0))
        y = float(prediction.get("y", 0))
        width = float(prediction.get("width", 0))
        height = float(prediction.get("height", 0))
        label = prediction.get("class") or prediction.get("class_name") or "object"
        label = normalize_label(label)
        confidence = float(prediction.get("confidence", prediction.get("conf", 0.0)))

        left = x - width / 2
        top = y - height / 2
        right = x + width / 2
        bottom = y + height / 2

        draw.rectangle((left, top, right, bottom), outline="#d95d39", width=3)
        draw.text(
            (left + 4, max(0, top - 14)),
            f"{label} {confidence:.2f}",
            fill="#d95d39",
            font=font,
        )

    image.save(output_path)


def detect_with_roboflow(output_path: Path):
    missing = []
    if not ROBOFLOW_API_KEY:
        missing.append("ROBOFLOW_API_KEY")
    if not ROBOFLOW_WORKSPACE:
        missing.append("ROBOFLOW_WORKSPACE")
    if not ROBOFLOW_WORKFLOW_ID:
        missing.append("ROBOFLOW_WORKFLOW_ID")
    if missing:
        raise RuntimeError("Missing Roboflow configuration: " + ", ".join(missing))

    from inference_sdk import InferenceHTTPClient

    client = InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=ROBOFLOW_API_KEY)
    result = client.run_workflow(
        workspace_name=ROBOFLOW_WORKSPACE,
        workflow_id=ROBOFLOW_WORKFLOW_ID,
        images={"image": str(output_path)},
        parameters={"classes": ROBOFLOW_CLASSES},
        use_cache=True,
    )

    predictions = extract_predictions(result)
    draw_predictions(output_path, predictions)
    return predictions


def run_detection(output_path: Path):
    if DETECTION_BACKEND == "roboflow":
        return detect_with_roboflow(output_path)
    return detect_with_yolo(output_path)


@app.route("/")
def home():
    error = session.pop("error", None) or MODEL_LOAD_ERROR
    result = session.pop("result", None)

    response = make_response(render_template(
        "index.html",
        backend_name=get_backend_label(),
        error=error,
        supported_classes=get_supported_classes(),
        yolo_conf=YOLO_CONF,
        yolo_imgsz=YOLO_IMGSZ,
        image=result["image"] if result else None,
        filename=result["filename"] if result else None,
        detections=result["detections"] if result else None,
    ))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "-1"
    return response


@app.route("/predict", methods=["POST"])
@app.route("/upload", methods=["POST"])
def predict():
    image = request.files.get("image") or request.files.get("file")

    if image is None or image.filename == "":
        session["error"] = "Select an image file first."
        return redirect(url_for("home"))

    if not allowed_file(image.filename):
        session["error"] = "Unsupported file type. Use JPG, JPEG, PNG, BMP, or WEBP."
        return redirect(url_for("home"))

    original_name = secure_filename(image.filename)
    suffix = Path(original_name).suffix.lower()
    output_name = f"{Path(original_name).stem}-{uuid4().hex[:8]}{suffix}"
    output_path = UPLOAD_DIR / output_name
    image.save(output_path)

    try:
        if DETECTION_BACKEND == "roboflow":
            detections = detect_with_roboflow(output_path)
        else:
            detections = detect_with_yolo(output_path)

        # SINGLE WEAPON MODE: Pick only the best one
        if SINGLE_WEAPON_MODE and detections:
            # Sort by confidence and take the best
            detections = [sorted(detections, key=lambda x: x["confidence"], reverse=True)[0]]
            draw_predictions(output_path, detections)
            
        if "text/html" in request.headers.get("Accept", ""):
            ui_detections = [{"class": d["class"], "confidence": d["confidence"], "bbox": d["bbox"]} for d in detections]
            session["result"] = {
                "image": f"uploads/{output_name}",
                "filename": original_name,
                "detections": ui_detections
            }
            return redirect(url_for("home"))
        else:
            return jsonify(detections)

    except Exception as e:
        session["error"] = str(e)
        return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
