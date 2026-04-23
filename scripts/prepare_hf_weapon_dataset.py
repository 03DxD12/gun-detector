import argparse
import shutil
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = BASE_DIR / "prepared_dataset" / "weapon_detector"
DATASET_ID = "Subh775/WeaponDetection_Grouped"
TARGET_CLASS = "GUN"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and convert the Hugging Face WeaponDetection_Grouped dataset to YOLO format."
    )
    parser.add_argument(
        "--dataset-id",
        default=DATASET_ID,
        help="Hugging Face dataset id to load.",
    )
    return parser.parse_args()


def clear_output(root: Path):
    if root.exists():
        shutil.rmtree(root)
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def sanitize_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def to_yolo_line(bbox, width: int, height: int) -> str:
    x_min, y_min, x_max, y_max = bbox
    x_min = min(max(float(x_min), 0.0), float(width))
    y_min = min(max(float(y_min), 0.0), float(height))
    x_max = min(max(float(x_max), 0.0), float(width))
    y_max = min(max(float(y_max), 0.0), float(height))
    box_w = max(0.0, x_max - x_min)
    box_h = max(0.0, y_max - y_min)
    if box_w == 0 or box_h == 0:
        return ""
    x_center = (x_min + box_w / 2) / width
    y_center = (y_min + box_h / 2) / height
    norm_w = box_w / width
    norm_h = box_h / height
    return f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"


def main():
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the 'datasets' package first: pip install datasets") from exc

    dataset = load_dataset(args.dataset_id)
    category_feature = dataset["train"].features["objects"]["category"]
    id_to_name = {idx: name for idx, name in enumerate(category_feature.feature.names)}

    clear_output(OUTPUT_ROOT)

    counts = {}
    for split_name, split_dataset in dataset.items():
        output_split = "val" if split_name == "validation" else split_name
        written = 0
        for row in split_dataset:
            width = int(row["width"])
            height = int(row["height"])
            categories = row["objects"]["category"]
            bboxes = row["objects"]["bbox"]

            label_lines = []
            for category_id, bbox in zip(categories, bboxes):
                if id_to_name[int(category_id)] != TARGET_CLASS:
                    continue
                line = to_yolo_line(bbox, width, height)
                if line:
                    label_lines.append(line)

            if not label_lines:
                continue

            image = row["image"].convert("RGB")
            suffix = ".jpg"
            stem = sanitize_stem(f"{split_name}_{row['image_id']}")
            image_path = OUTPUT_ROOT / "images" / output_split / f"{stem}{suffix}"
            label_path = OUTPUT_ROOT / "labels" / output_split / f"{stem}.txt"

            image.save(image_path, quality=95)
            label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
            written += 1

        counts[output_split] = written

    yaml_content = "\n".join(
        [
            f"path: '{OUTPUT_ROOT.as_posix()}'",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            "names:",
            "  0: weapon",
            "",
        ]
    )
    (BASE_DIR / "dataset.weapon_detector.yaml").write_text(yaml_content, encoding="utf-8")
    (BASE_DIR / "dataset.yaml").write_text(yaml_content, encoding="utf-8")

    print("Prepared Hugging Face dataset:", args.dataset_id)
    print("Output root:", OUTPUT_ROOT)
    print("train", counts.get("train", 0))
    print("val", counts.get("val", 0))
    print("test", counts.get("test", 0))


if __name__ == "__main__":
    main()
