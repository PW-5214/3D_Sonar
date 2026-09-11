from ultralytics import YOLO
import torch

DATA = r"C:\Users\Admin\Downloads\SIH2026\SONAR_DATASET\data.yaml"

print("=" * 80)
print("YOLO26m SONAR OBJECT DETECTION TRAINING")
print("=" * 80)

print("PyTorch :", torch.__version__)
print("CUDA    :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU     :", torch.cuda.get_device_name(0))
    print(
        "VRAM    :",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2
        ),
        "GB"
    )

print("=" * 80)

model = YOLO("yolo26m.pt")

results = model.train(
    data=DATA,

    # Training
    epochs=120,
    imgsz=640,
    batch=16,

    # GPU
    device=0,

    # Windows
    workers=0,

    # Memory
    cache=False,
    amp=True,

    # Optimizer
    optimizer="auto",

    # Validation
    val=True,

    # Saving
    project=r"C:\Users\Admin\Downloads\SIH2026\runs",
    name="sonar_yolo26m",
    exist_ok=False,
    save=True,
    save_period=10,

    # Early stopping
    patience=25,

    # Plots / metrics
    plots=True,

    # Sonar-friendly augmentation
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.15,

    degrees=0.0,
    translate=0.05,
    scale=0.20,
    shear=0.0,
    perspective=0.0,

    flipud=0.0,
    fliplr=0.5,

    mosaic=0.5,
    mixup=0.0,

    # Reproducibility
    seed=42,
    deterministic=True
)

print("=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(results)