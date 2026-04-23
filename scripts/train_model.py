import argparse
from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PROJECT_DIR = BASE_DIR / "runs" / "detect"


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on a prepared dataset preset.")
    parser.add_argument(
        "--preset",
        choices=["weapon_detector", "pistol_detection", "gun_types"],
        default="weapon_detector",
        help="Prepared dataset preset to train.",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="Base model checkpoint.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--project",
        default=str(DEFAULT_PROJECT_DIR),
        help="Training output root.",
    )
    parser.add_argument("--name", default=None, help="Training run name.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_yaml = BASE_DIR / f"dataset.{args.preset}.yaml"
    run_name = args.name or args.preset

    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"{dataset_yaml.name} not found. Run scripts/prepare_dataset.py --preset {args.preset} first."
        )

    model = YOLO(args.model)
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=run_name,
    )


if __name__ == "__main__":
    main()
