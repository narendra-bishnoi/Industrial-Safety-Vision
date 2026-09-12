"""Converts the Kaggle "Safety Helmet Detection" dataset (Pascal VOC XML)
into the YOLO training format expected by scripts/train_ppe_model.py.

Source: https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection
License: CC0 1.0 (public domain)
Classes in the source annotations: helmet, head, person

Usage:
    python -m scripts.prepare_dataset --source data/raw --dest data/yolo --val-split 0.15

Expected --source layout (as the Kaggle zip extracts):
    data/raw/images/*.png
    data/raw/annotations/*.xml
"""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = ["helmet", "head", "person"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}


def _convert_annotation(xml_path: Path) -> tuple[str, list[str]] | None:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = float(size.findtext("width"))
    img_h = float(size.findtext("height"))
    filename = root.findtext("filename")
    if not filename or img_w <= 0 or img_h <= 0:
        return None

    lines = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in CLASS_TO_ID:
            continue
        box = obj.find("bndbox")
        xmin, ymin = float(box.findtext("xmin")), float(box.findtext("ymin"))
        xmax, ymax = float(box.findtext("xmax")), float(box.findtext("ymax"))

        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h
        lines.append(f"{CLASS_TO_ID[name]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return filename, lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/raw", help="Folder containing images/ and annotations/")
    parser.add_argument("--dest", default="data/yolo", help="Output folder for YOLO-format dataset")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.source)
    images_dir = source / "images"
    ann_dir = source / "annotations"
    if not images_dir.is_dir() or not ann_dir.is_dir():
        raise SystemExit(
            f"Expected {images_dir} and {ann_dir} to exist. "
            "Extract the Kaggle zip into data/raw/ first (see data/README.md)."
        )

    xml_files = sorted(ann_dir.glob("*.xml"))
    if not xml_files:
        raise SystemExit(f"No annotation XML files found in {ann_dir}")

    random.Random(args.seed).shuffle(xml_files)
    n_val = int(len(xml_files) * args.val_split)
    splits = {"val": xml_files[:n_val], "train": xml_files[n_val:]}

    dest = Path(args.dest)
    for split, files in splits.items():
        (dest / "images" / split).mkdir(parents=True, exist_ok=True)
        (dest / "labels" / split).mkdir(parents=True, exist_ok=True)

        written = 0
        for xml_path in files:
            converted = _convert_annotation(xml_path)
            if converted is None:
                continue
            filename, lines = converted
            src_image = images_dir / filename
            if not src_image.is_file():
                continue
            shutil.copy(src_image, dest / "images" / split / filename)
            label_name = Path(filename).with_suffix(".txt").name
            (dest / "labels" / split / label_name).write_text("\n".join(lines), encoding="utf-8")
            written += 1
        print(f"{split}: wrote {written} image/label pairs")

    data_yaml = dest / "data.yaml"
    data_yaml.write_text(
        "path: " + str(dest.resolve()).replace("\\", "/") + "\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names: {CLASSES}\n",
        encoding="utf-8",
    )
    print(f"Wrote {data_yaml}")


if __name__ == "__main__":
    main()
