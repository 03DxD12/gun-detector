import argparse
import csv
import json
import random
import shutil
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_ROOT = BASE_DIR / "dataset" / "raw"
PREPARED_ROOT = BASE_DIR / "prepared_dataset"

WEAPON_SOURCE_ROOT = RAW_ROOT / "kaggle_weapon_detection"
WEAPON_IMAGES_ROOT = WEAPON_SOURCE_ROOT / "weapon_detection"
WEAPON_METADATA_PATH = WEAPON_SOURCE_ROOT / "metadata.csv"

GUN_DETECTION_ROOT = RAW_ROOT / "kaggle_gun_detection" / "flat_yolo"
ROBOFLOW_PISTOLS_ROOT = RAW_ROOT / "roboflow_pistols" / "coco_export"
ROBOFLOW_COCO_PATH = ROBOFLOW_PISTOLS_ROOT / "_annotations.coco.json"

GUN_TYPE_CLASSES = [
    "Automatic Rifle",
    "Handgun",
    "Shotgun",
    "SMG",
    "Sniper",
]

PRESET_CONFIG = {
    "weapon_detector": {
        "output_dir": PREPARED_ROOT / "weapon_detector",
        "yaml_path": BASE_DIR / "dataset.weapon_detector.yaml",
        "classes": ["weapon"],
    },
    "gun_types": {
        "output_dir": PREPARED_ROOT / "gun_types",
        "yaml_path": BASE_DIR / "dataset.gun_types.yaml",
        "classes": [name.lower().replace(" ", "_") for name in GUN_TYPE_CLASSES],
    },
    "pistol_detection": {
        "output_dir": PREPARED_ROOT / "pistol_detection",
        "yaml_path": BASE_DIR / "dataset.pistol_detection.yaml",
        "classes": ["pistol"],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare train/val/test YOLO datasets from the organized raw sources."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_CONFIG),
        default="pistol_detection",
        help="Dataset preset to prepare.",
    )
    parser.add_argument("--train", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--test", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def clear_output(root: Path):
    if root.exists():
        shutil.rmtree(root)

    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def split_samples(samples, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError("Train, val, and test ratios must add up to 1.0")

    samples = list(samples)
    rng = random.Random(seed)
    rng.shuffle(samples)

    train_end = int(len(samples) * train_ratio)
    val_end = train_end + int(len(samples) * val_ratio)

    return {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }


def write_dataset_yaml(dataset_root: Path, yaml_path: Path, classes):
    lines = [
        f"path: '{dataset_root.as_posix()}'",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for class_id, class_name in enumerate(classes):
        lines.append(f"  {class_id}: {class_name}")

    content = "\n".join(lines) + "\n"
    yaml_path.write_text(content, encoding="utf-8")
    (BASE_DIR / "dataset.yaml").write_text(content, encoding="utf-8")


def copy_samples(split_map, output_root: Path):
    for split_name, items in split_map.items():
        for item in items:
            image_target = output_root / "images" / split_name / item["output_stem"]
            label_target = output_root / "labels" / split_name / f"{Path(item['output_stem']).stem}.txt"
            shutil.copy2(item["image_path"], image_target)
            label_target.write_text(item["label_text"], encoding="utf-8")


def find_weapon_source_file(filename: str, folder: str) -> Path | None:
    for split in ("train", "val"):
        candidate = WEAPON_IMAGES_ROOT / split / folder / filename
        if candidate.exists():
            return candidate
    return None


def load_weapon_type_samples():
    class_to_id = {name: idx for idx, name in enumerate(GUN_TYPE_CLASSES)}
    samples = []

    with WEAPON_METADATA_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            class_name = row["imagefile"].rsplit("_", 1)[0]
            if class_name not in class_to_id:
                continue

            image_path = find_weapon_source_file(row["imagefile"], "images")
            label_path = find_weapon_source_file(row["labelfile"], "labels")
            if image_path is None or label_path is None:
                continue

            label_lines = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 5:
                    label_lines.append(" ".join([str(class_to_id[class_name]), *parts[1:]]))

            if not label_lines:
                continue

            samples.append(
                {
                    "image_path": image_path,
                    "output_stem": f"weapon__{image_path.name}",
                    "label_text": "\n".join(label_lines) + "\n",
                }
            )

    return samples


def load_flat_yolo_samples(root: Path, source_prefix: str, class_id: int):
    image_paths = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        image_paths.extend(root.glob(pattern))

    samples = []
    for image_path in image_paths:
        label_path = root / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        label_lines = []
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) == 5:
                label_lines.append(" ".join([str(class_id), *parts[1:]]))

        if not label_lines:
            continue

        samples.append(
            {
                "image_path": image_path,
                "output_stem": f"{source_prefix}__{image_path.name}",
                "label_text": "\n".join(label_lines) + "\n",
            }
        )

    return samples


def coco_bbox_to_yolo_line(annotation, image_meta, class_id: int):
    x, y, width, height = annotation["bbox"]
    image_width = image_meta["width"]
    image_height = image_meta["height"]

    x_center = (x + width / 2) / image_width
    y_center = (y + height / 2) / image_height
    norm_width = width / image_width
    norm_height = height / image_height

    return f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"


def load_roboflow_coco_samples(root: Path, annotations_path: Path, source_prefix: str, class_id: int):
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    images_by_id = {item["id"]: item for item in data.get("images", [])}
    annotations_by_image = {}

    for annotation in data.get("annotations", []):
        if annotation.get("image_id") not in images_by_id:
            continue
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    samples = []
    for image_id, annotations in annotations_by_image.items():
        image_meta = images_by_id[image_id]
        image_path = root / image_meta["file_name"]
        if not image_path.exists():
            continue

        label_lines = [coco_bbox_to_yolo_line(item, image_meta, class_id) for item in annotations]
        if not label_lines:
            continue

        samples.append(
            {
                "image_path": image_path,
                "output_stem": f"{source_prefix}__{image_path.name}",
                "label_text": "\n".join(label_lines) + "\n",
            }
        )

    return samples


def build_samples_for_preset(preset: str):
    if preset == "weapon_detector":
        grouped_root = PREPARED_ROOT / "weapon_detector"
        if not grouped_root.exists():
            raise RuntimeError(
                "prepared_dataset/weapon_detector not found. Run scripts/prepare_hf_weapon_dataset.py first."
            )

        samples = []
        for split in ("train", "val", "test"):
            images_dir = grouped_root / "images" / split
            labels_dir = grouped_root / "labels" / split
            for image_path in images_dir.iterdir():
                if not image_path.is_file():
                    continue
                label_path = labels_dir / f"{image_path.stem}.txt"
                if not label_path.exists():
                    continue
                samples.append(
                    {
                        "image_path": image_path,
                        "output_stem": image_path.name,
                        "label_text": label_path.read_text(encoding="utf-8"),
                    }
                )
        return samples

    if preset == "gun_types":
        return load_weapon_type_samples()

    if preset == "pistol_detection":
        samples = []
        samples.extend(load_flat_yolo_samples(GUN_DETECTION_ROOT, "gun_detection", 0))
        samples.extend(
            load_roboflow_coco_samples(
                ROBOFLOW_PISTOLS_ROOT,
                ROBOFLOW_COCO_PATH,
                "roboflow_pistols",
                0,
            )
        )
        return samples

    raise ValueError(f"Unsupported preset: {preset}")


def main():
    args = parse_args()
    config = PRESET_CONFIG[args.preset]
    samples = build_samples_for_preset(args.preset)

    if not samples:
        raise RuntimeError(f"No samples found for preset '{args.preset}'.")

    clear_output(config["output_dir"])
    split_map = split_samples(samples, args.train, args.val, args.test, args.seed)
    copy_samples(split_map, config["output_dir"])
    write_dataset_yaml(config["output_dir"], config["yaml_path"], config["classes"])

    print(f"Preset: {args.preset}")
    print("Prepared dataset written to:", config["output_dir"])
    print("Dataset YAML:", config["yaml_path"])
    for split_name, items in split_map.items():
        print(f"{split_name}: {len(items)} images")
    print("classes:", ", ".join(config["classes"]))


if __name__ == "__main__":
    main()
