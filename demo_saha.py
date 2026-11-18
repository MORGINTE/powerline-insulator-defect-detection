"""Automated Power Line Insulator Defect Detection Inference Pipeline.

Runs YOLOv8 inference on inspection images, applies class-specific confidence
thresholds for ``broken`` and ``pollution-flashover`` defects, categorizes
images by severity (CRITICAL / WARNING / CLEAN), saves annotated images, and
exports a prioritized summary CSV report.
"""

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "deneyC_960" / "weights" / "best.pt"
DEFAULT_SOURCE = PROJECT_ROOT / "yolo_deneyA" / "images" / "test"
DEFAULT_LABELS = PROJECT_ROOT / "yolo_deneyA" / "labels" / "test"
DEFAULT_OUTPUT = PROJECT_ROOT / "demo_cikti"

# Class IDs matching yolo_deneyA/data.yaml
FLASHOVER_ID = 0
BROKEN_ID = 1
INSULATOR_ID = 2

SEVERITY_ORDER: Dict[str, int] = {"CRITICAL": 0, "WARNING": 1, "CLEAN": 2}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def has_ground_truth_defect(image_path: Path, labels_dir: Optional[Path]) -> Optional[bool]:
    """Check whether a ground-truth YOLO label file contains any defect class.

    Args:
        image_path: Path to the input image file.
        labels_dir: Directory containing YOLO ``.txt`` label files, or ``None``.

    Returns:
        ``True`` if a ``broken`` or ``pollution-flashover`` annotation exists,
        ``False`` if the label file exists with no defect annotations, or
        ``None`` if no corresponding label file is found.
    """
    if labels_dir is None or not labels_dir.exists():
        return None
    label_path = labels_dir / f"{image_path.stem}.txt"
    if not label_path.exists():
        return None

    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if parts and int(parts[0]) in (FLASHOVER_ID, BROKEN_ID):
            return True
    return False


def filter_detections(
    boxes_data: torch.Tensor,
    broken_conf: float,
    flashover_conf: float,
) -> torch.Tensor:
    """Filter raw YOLO detections using class-specific confidence thresholds.

    Args:
        boxes_data: Tensor of shape ``(N, 6)`` with ``[x1, y1, x2, y2, conf, cls]``.
        broken_conf: Minimum confidence threshold for the ``broken`` class.
        flashover_conf: Minimum confidence threshold for ``pollution-flashover``.

    Returns:
        Filtered tensor containing only detections that satisfy per-class thresholds.
    """
    if boxes_data.numel() == 0:
        return boxes_data

    class_ids = boxes_data[:, 5]
    confidences = boxes_data[:, 4]
    keep_mask = (
        ((class_ids == BROKEN_ID) & (confidences >= broken_conf))
        | ((class_ids == FLASHOVER_ID) & (confidences >= flashover_conf))
        | (class_ids == INSULATOR_ID)
    )
    return boxes_data[keep_mask]


def classify_image_severity(
    class_ids: List[int],
    confidences: List[float],
) -> Tuple[str, float, int, int]:
    """Determine image-level inspection severity from filtered detections.

    Args:
        class_ids: Predicted class IDs after threshold filtering.
        confidences: Corresponding confidence scores.

    Returns:
        Tuple of ``(severity, max_confidence, broken_count, flashover_count)``.
    """
    broken_scores = [c for cls_id, c in zip(class_ids, confidences) if cls_id == BROKEN_ID]
    flashover_scores = [c for cls_id, c in zip(class_ids, confidences) if cls_id == FLASHOVER_ID]

    if broken_scores:
        severity = "CRITICAL"
        max_score = max(broken_scores)
    elif flashover_scores:
        severity = "WARNING"
        max_score = max(flashover_scores)
    else:
        severity = "CLEAN"
        max_score = 0.0

    return severity, max_score, len(broken_scores), len(flashover_scores)


def export_summary_csv(records: List[Dict], csv_path: Path) -> None:
    """Sort inspection records by severity/confidence and write them to CSV.

    Args:
        records: List of per-image inspection result dictionaries.
        csv_path: Destination path for the summary CSV file.
    """
    records.sort(key=lambda r: (SEVERITY_ORDER[r["severity"]], -r["max_confidence"]))
    gt_map = {True: "yes", False: "no", None: "N/A"}

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "rank",
            "filename",
            "severity",
            "max_confidence",
            "broken_count",
            "flashover_count",
            "ground_truth_defect",
        ])
        for idx, rec in enumerate(records, start=1):
            writer.writerow([
                idx,
                rec["filename"],
                rec["severity"],
                rec["max_confidence"],
                rec["broken_count"],
                rec["flashover_count"],
                gt_map[rec["ground_truth"]],
            ])


def print_evaluation_summary(records: List[Dict], output_dir: Path) -> None:
    """Print aggregate detection counts and optional ground-truth recall metrics."""
    counts = {lvl: sum(r["severity"] == lvl for r in records) for lvl in SEVERITY_ORDER}
    print(f"Processed images: {len(records)} | Severity breakdown: {counts}")

    evaluated = [r for r in records if r["ground_truth"] is not None]
    if evaluated:
        tp = sum(bool(r["ground_truth"]) and r["severity"] != "CLEAN" for r in evaluated)
        fn = sum(bool(r["ground_truth"]) and r["severity"] == "CLEAN" for r in evaluated)
        fp = sum((not r["ground_truth"]) and r["severity"] != "CLEAN" for r in evaluated)
        total_defective = tp + fn
        print(f"Ground-truth defective images: {total_defective}")
        print(f"  Flagged (TP): {tp} | Missed (FN): {fn} | False Alarms (FP): {fp}")
        if total_defective:
            print(f"  Image-level recall: {100.0 * tp / total_defective:.1f}%")

    print(f"Output directory: {output_dir}")


def run_inference(
    weights_path: Path,
    source_paths: List[Path],
    output_dir: Path,
    labels_dir: Optional[Path] = None,
    imgsz: int = 960,
    base_conf: float = 0.25,
    broken_conf: float = 0.50,
    flashover_conf: float = 0.25,
    iou: float = 0.60,
    device: Optional[str] = None,
    flat_output: bool = False,
) -> List[Dict]:
    """Run YOLOv8 defect detection on a list of images and save annotated outputs.

    Args:
        weights_path: Path to the trained YOLOv8 ``.pt`` weights file.
        source_paths: List of image paths to process.
        output_dir: Directory where annotated images and summary CSV are saved.
        labels_dir: Optional directory with YOLO ground-truth ``.txt`` labels.
        imgsz: Inference image resolution.
        base_conf: Base confidence threshold passed to ``model.predict``.
        broken_conf: Post-filter confidence threshold for ``broken`` defects.
        flashover_conf: Post-filter confidence threshold for ``pollution-flashover``.
        iou: NMS IoU threshold.
        device: Inference device (e.g. ``"0"`` or ``"cpu"``). Auto-detected if ``None``.
        flat_output: If ``True``, save annotated images directly in ``output_dir``
            instead of severity subdirectories.

    Returns:
        List of per-image inspection result dictionaries.
    """
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    if not source_paths:
        raise ValueError("No valid input images found.")

    if not flat_output:
        shutil.rmtree(output_dir, ignore_errors=True)
        for level in SEVERITY_ORDER:
            (output_dir / level).mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = device if device is not None else ("0" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(weights_path))
    records: List[Dict] = []

    results = model.predict(
        source=[str(p) for p in source_paths],
        imgsz=imgsz,
        conf=base_conf,
        iou=iou,
        device=resolved_device,
        stream=True,
        verbose=False,
    )

    for result in results:
        filtered_data = filter_detections(
            result.boxes.data,
            broken_conf=broken_conf,
            flashover_conf=flashover_conf,
        )
        result.boxes = Boxes(filtered_data, result.orig_shape)

        class_ids = [int(c) for c in result.boxes.cls.tolist()]
        confidences = result.boxes.conf.tolist()
        severity, max_score, broken_cnt, flashover_cnt = classify_image_severity(
            class_ids, confidences
        )

        img_path = Path(result.path)
        save_dir = output_dir if flat_output else (output_dir / severity)
        Image.fromarray(result.plot()[..., ::-1]).save(save_dir / f"{img_path.stem}.jpg")

        records.append({
            "filename": img_path.name,
            "severity": severity,
            "max_confidence": round(max_score, 3),
            "broken_count": broken_cnt,
            "flashover_count": flashover_cnt,
            "ground_truth": has_ground_truth_defect(img_path, labels_dir),
        })

    export_summary_csv(records, output_dir / "summary.csv")
    print_evaluation_summary(records, output_dir)
    return records


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the inference pipeline."""
    parser = argparse.ArgumentParser(
        description="Automated Power Line Insulator Defect Detection (YOLOv8)"
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to YOLOv8 .pt weights")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Input image file or directory")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="Optional YOLO labels directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory for results")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Base confidence threshold")
    parser.add_argument("--broken-conf", type=float, default=0.50, help="Confidence threshold for 'broken'")
    parser.add_argument("--flashover-conf", type=float, default=0.25, help="Confidence threshold for 'pollution-flashover'")
    parser.add_argument("--iou", type=float, default=0.60, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default=None, help="Device ('0', 'cpu', etc.)")
    parser.add_argument("--flat-output", action="store_true", help="Save images directly into output dir")
    return parser.parse_args()


def collect_images(source: Path) -> List[Path]:
    """Collect image paths from a single file or a directory."""
    if source.is_file():
        return [source] if source.suffix.lower() in IMAGE_EXTENSIONS else []
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    raise FileNotFoundError(f"Source path does not exist: {source}")


def main() -> None:
    """Entry point for CLI execution."""
    args = parse_args()
    images = collect_images(args.source)
    run_inference(
        weights_path=args.weights,
        source_paths=images,
        output_dir=args.output,
        labels_dir=args.labels,
        imgsz=args.imgsz,
        base_conf=args.conf,
        broken_conf=args.broken_conf,
        flashover_conf=args.flashover_conf,
        iou=args.iou,
        device=args.device,
        flat_output=args.flat_output,
    )


if __name__ == "__main__":
    main()
