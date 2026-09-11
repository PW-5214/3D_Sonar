# AbyssScan 3D — Side-Scan Sonar Debris Dashboard

A Streamlit web prototype built around the supplied **YOLO26m SSS models**.

## Included trained classes

The uploaded checkpoint metadata contains four classes:

- `shipwreck`
- `pipeline_cable`
- `ghost_net`
- `submarine_pipeline`

## Included checkpoints

- `models/yolo_enhanced_768_best.pt` — trained by `train_sonar.py`, 768 px input, sonar-oriented preprocessing/augmentation.
- `models/yolo_standard_640_best.pt` — trained by `11_train_yolo26.py`, 640 px input.

The checkpoints were created with **Ultralytics 8.4.146**; that exact version is pinned in `requirements.txt`.

## Run on Windows

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Or double-click `run_windows.bat` after Python 3.11/3.12 and pip are installed.

## What the app does

1. Upload PNG/JPG/TIFF SSS imagery.
2. Run your YOLO model.
3. Detect nadir/water-column location.
4. Convert cross-track pixels to metric ground range when sonar range/altitude are supplied.
5. Search outward from each target for an acoustic shadow.
6. Estimate target height from shadow geometry when conditions are suitable.
7. Build an interactive Plotly 3D scene that **does not map darkness to depth**.
8. Export JSON/CSV contact reports.
9. Optionally geotag contacts from a towfish/AUV position and heading.

## Important technical distinction

Conventional SSS intensity is backscatter, not bathymetry. The app therefore has two effective 3D states:

**Qualitative 3D** — if only an image is available. Positive target structure inside model detections is visualized as relief; shadows stay at the seabed.

**Physics-assisted local 3D** — if range and sonar altitude are supplied and a usable acoustic shadow is found. The target ROI is scaled using a shadow-height constraint:

```text
h = H * L / (x + L)
```

where `H` is sonar altitude, `x` is target ground range, and `L` is ground shadow length.

For engineering-grade 3D, add raw XTF/JSF decoding, pose/navigation correction, and ideally MBES bathymetry.

## Recommended next upgrade

The next development step should ingest raw **XTF/JSF**, preserve ping metadata, perform water-column/slant-range correction before inference, and use MBES as the actual Z surface while SSS supplies high-resolution acoustic texture and object evidence.
