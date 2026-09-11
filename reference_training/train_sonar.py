from pathlib import Path
import multiprocessing
import torch
from ultralytics import YOLO


BASE_DIR = Path(r"C:\Users\Admin\Downloads\SIH2026")
DATASET_DIR = BASE_DIR / "SONAR_ENHANCED_TILED"
YAML_PATH = DATASET_DIR / "sonar.yaml"
RUNS_DIR = BASE_DIR / "runs"


def main():

    print("=" * 60)
    print("GPU CHECK")
    print("=" * 60)

    print("CUDA Available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected.")

    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU Memory:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB"
    )

    print("=" * 60)
    print("DATASET CHECK")
    print("=" * 60)

    print("YAML:", YAML_PATH)
    print("Exists:", YAML_PATH.exists())

    if not YAML_PATH.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {YAML_PATH}"
        )

    print("=" * 60)
    print("LOADING YOLO26m")
    print("=" * 60)

    model = YOLO("yolo26m.pt")

    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    results = model.train(
        data=str(YAML_PATH),

        epochs=60,
        imgsz=768,
        batch=8,

        device=0,
        workers=4,

        patience=15,

        pretrained=True,
        optimizer="AdamW",

        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,

        warmup_epochs=5,
        cos_lr=True,

        hsv_h=0.0,
        hsv_s=0.15,
        hsv_v=0.20,

        degrees=5.0,
        translate=0.05,
        scale=0.20,

        shear=0.0,
        perspective=0.0,

        flipud=0.0,
        fliplr=0.5,

        mosaic=0.5,
        mixup=0.05,
        copy_paste=0.0,

        close_mosaic=10,

        project=str(RUNS_DIR),
        name="sonar_yolo_enhanced_768",

        exist_ok=True,
        plots=True,
        save=True,
        verbose=True
    )

    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(
        "Best model:",
        BASE_DIR /
        "runs" /
        "sonar_yolo_enhanced_768" /
        "weights" /
        "best.pt"
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()