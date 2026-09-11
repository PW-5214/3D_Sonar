# Model information extracted from uploaded files

The supplied archive contains two YOLO26m training configurations and four checkpoints (`best.pt`/`last.pt` for each run). This app bundles only the two `best.pt` files.

## Standard run

Source: `reference_training/11_train_yolo26.py`

- base: `yolo26m.pt`
- epochs: 120
- input: 640
- batch: 16
- augmentations: brightness, translate, scale, horizontal flip, mosaic
- run name: `sonar_yolo26m`

## Enhanced run

Source: `reference_training/train_sonar.py`

- base: `yolo26m.pt`
- epochs: 60
- input: 768
- batch: 8
- optimizer: AdamW
- cosine LR
- sonar-oriented augmentation
- run name: `sonar_yolo_enhanced_768`

## Checkpoint class names

Extracted from the uploaded checkpoint metadata:

```text
shipwreck
pipeline_cable
ghost_net
submarine_pipeline
```

The serialized checkpoint also contains the Ultralytics version string `8.4.146`.
