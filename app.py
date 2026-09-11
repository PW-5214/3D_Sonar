from __future__ import annotations

import json
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from sonar3d.inference import load_yolo, predict, draw_detections
from sonar3d.geometry import (
    build_detection_aware_relief,
    detect_nadir_column,
    estimate_shadow_height,
    metric_dimensions,
    signed_ground_axis,
)
from sonar3d.georef import geotag_detection
from sonar3d.visualization import surface_figure


APP_DIR = Path(__file__).resolve().parent
MODELS = {
    "Enhanced 768 (recommended)": APP_DIR / "models" / "yolo_enhanced_768_best.pt",
    "Standard 640": APP_DIR / "models" / "yolo_standard_640_best.pt",
}

st.set_page_config(
    page_title="AbyssScan | SSS Debris Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root { --navy:#071018; --panel:#0b1821; --line:#1f3541; --cyan:#4fd1c5; --gold:#d9a441; }
.stApp { background: #071018; color: #e6eef3; }
[data-testid="stSidebar"] { background: #09141d; border-right: 1px solid #1c303b; }
[data-testid="stMetric"] { background:#0b1821; border:1px solid #1d3440; border-radius:12px; padding:12px; }
.block-container { padding-top: 1.6rem; }
.sss-title { font-size:2.1rem; font-weight:750; letter-spacing:-0.03em; margin-bottom:.1rem; }
.sss-kicker { color:#73ded4; font-size:.82rem; letter-spacing:.16em; text-transform:uppercase; font-weight:700; }
.sss-muted { color:#93a8b5; }
.sss-card { background:#0b1821; border:1px solid #1d3440; border-radius:14px; padding:16px 18px; }
.badge { display:inline-block; padding:4px 9px; border-radius:999px; background:#10313a; color:#78e0d5; border:1px solid #20515b; font-size:.78rem; }
.warning-card { background:#20190a; border:1px solid #665019; border-radius:12px; padding:12px 14px; color:#f2d38d; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="sss-kicker">Side-Scan Sonar Decision Support</div>', unsafe_allow_html=True)
st.markdown('<div class="sss-title">AbyssScan 3D</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sss-muted">Detect anthropogenic sonar targets with your trained YOLO26m model, estimate geometry from acoustic shadows when metadata is available, and inspect each contact in an interactive 3D scene.</div>',
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_model(path: str):
    return load_yolo(path)


with st.sidebar:
    st.subheader("Inference")
    model_label = st.selectbox("Model", list(MODELS.keys()), index=0)
    model_path = MODELS[model_label]
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.30, 0.05)
    iou = st.slider("NMS IoU", 0.10, 0.90, 0.45, 0.05)
    imgsz = 768 if "768" in model_label else 640
    st.caption(f"Input size: {imgsz}px • {model_path.name}")

    st.divider()
    st.subheader("Sonar geometry")
    auto_nadir = st.checkbox("Auto-detect nadir", value=True)
    range_mode = st.selectbox("Image range geometry", ["Slant range", "Ground range corrected"], index=0)
    max_range = st.number_input("Range at image edge (m)", min_value=1.0, value=75.0, step=5.0)
    altitude = st.number_input("Sonar altitude above seabed (m)", min_value=0.1, value=10.0, step=0.5)
    along_res = st.number_input("Along-track metres / pixel", min_value=0.0, value=0.0, step=0.01, format="%.3f")
    st.caption("Set metres/pixel to 0 if unknown. Metric dimensions will remain partial rather than guessed.")

    st.divider()
    st.subheader("Optional georeference")
    enable_geo = st.checkbox("Geotag contacts", value=False)
    if enable_geo:
        lat = st.number_input("Towfish/AUV latitude", value=18.520400, format="%.6f")
        lon = st.number_input("Towfish/AUV longitude", value=73.856700, format="%.6f")
        heading = st.number_input("Heading (° true)", min_value=0.0, max_value=359.99, value=0.0, step=1.0)
        st.caption("Use towfish/AUV position or a layback-corrected position—not vessel GPS alone.")
    else:
        lat = lon = heading = None

uploaded = st.file_uploader("Upload an SSS image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])

if uploaded is None:
    st.markdown(
        """
<div class="sss-card">
<b>Ready for your survey image.</b><br>
Upload a rendered SSS frame or waterfall image. The current prototype uses your four-class detector:
<code>shipwreck</code>, <code>pipeline_cable</code>, <code>ghost_net</code>, and <code>submarine_pipeline</code>.
<br><br>
For defensible metric 3D, provide range and altitude metadata. Without them, the app intentionally labels the scene as qualitative relief rather than measured bathymetry.
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()

image = Image.open(uploaded).convert("RGB")
gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
H, W = gray.shape

try:
    with st.spinner("Loading model and scanning sonar frame…"):
        model = get_model(str(model_path))
        detections = predict(model, image, confidence=conf, iou=iou, imgsz=imgsz)
except Exception as exc:
    st.error(f"Model inference failed: {exc}")
    st.info("Install the exact environment from requirements.txt. These checkpoints were created with Ultralytics 8.4.146.")
    st.stop()

nadir_auto_x, nadir_conf = detect_nadir_column(gray)
if auto_nadir:
    nadir_x = nadir_auto_x
else:
    nadir_x = st.sidebar.number_input("Nadir x-column", min_value=0, max_value=W - 1, value=W // 2, step=1)

range_mode_key = "slant" if range_mode.startswith("Slant") else "ground"
try:
    ground_axis = signed_ground_axis(W, int(nadir_x), float(max_range), float(altitude), range_mode_key)
except Exception:
    ground_axis = None

shadow_estimates = []
physical_heights = {}
if ground_axis is not None and altitude > 0:
    for det in detections:
        est = estimate_shadow_height(gray, det, int(nadir_x), ground_axis, float(altitude))
        shadow_estimates.append(est)
        if est.shadow_found and est.height_m is not None and est.quality >= 0.15:
            physical_heights[det.id] = est.height_m

z = build_detection_aware_relief(
    gray,
    detections,
    physical_heights=physical_heights,
    qualitative_height=max(0.8, float(altitude) * 0.08),
    seabed_texture=max(0.01, float(altitude) * 0.002),
)

annotated = draw_detections(image, detections)

# ---- Summary metrics ----
metric_cols = st.columns(5)
metric_cols[0].metric("Contacts", len(detections))
metric_cols[1].metric("Ghost nets", sum(d.class_name == "ghost_net" for d in detections))
metric_cols[2].metric("Mean confidence", f"{(np.mean([d.confidence for d in detections]) * 100 if detections else 0):.1f}%")
metric_cols[3].metric("Nadir", f"x={int(nadir_x)}", f"auto {nadir_conf*100:.0f}%" if auto_nadir else "manual")
metric_cols[4].metric("Shadow heights", f"{len(physical_heights)}/{len(detections)}")

tab_scan, tab_3d, tab_report, tab_method = st.tabs(["2D Detection", "Interactive 3D", "Contact Report", "Method & Limits"])

with tab_scan:
    c1, c2 = st.columns([1.75, 1], gap="large")
    with c1:
        st.image(annotated, caption="YOLO26m detections", use_container_width=True)
    with c2:
        st.markdown("#### Scan interpretation")
        st.markdown(f"**Image:** {W} × {H} px  ")
        st.markdown(f"**Estimated nadir:** x = {int(nadir_x)}  ")
        st.markdown(f"**Geometry mode:** {range_mode}  ")
        st.markdown(f"**Range:** {max_range:.1f} m per side  ")
        st.markdown(f"**Altitude:** {altitude:.1f} m  ")
        if detections:
            st.markdown("#### Detected classes")
            counts = pd.Series([d.class_name for d in detections]).value_counts().rename_axis("class").reset_index(name="count")
            st.dataframe(counts, hide_index=True, use_container_width=True)
        else:
            st.info("No contacts above the selected confidence threshold.")

with tab_3d:
    st.markdown(
        '<div class="warning-card"><b>Important:</b> SSS intensity is not depth. This viewer does not map black pixels into holes. Detected positive returns are elevated; dark acoustic shadows stay near the seabed. When a valid outward shadow can be paired with a target and altitude/range metadata is supplied, the ROI is scaled using shadow geometry. Otherwise the Z-axis is qualitative.</div>',
        unsafe_allow_html=True,
    )
    fig = surface_figure(
        gray,
        z,
        int(nadir_x),
        ground_axis_m=ground_axis,
        along_track_m_per_px=(float(along_res) if along_res > 0 else None),
        title="Detection-aware sonar relief — rotate, pan and zoom",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

    if detections:
        selected = st.selectbox("Inspect contact", [f"{d.id} • {d.class_name} • {d.confidence*100:.1f}%" for d in detections])
        selected_id = selected.split(" • ")[0]
        det = next(d for d in detections if d.id == selected_id)
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        pad_x, pad_y = max(10, int((x2-x1)*0.7)), max(10, int((y2-y1)*0.7))
        xa, xb = max(0, x1-pad_x), min(W, x2+pad_x)
        ya, yb = max(0, y1-pad_y), min(H, y2+pad_y)
        crop_gray = gray[ya:yb, xa:xb]
        crop_z = z[ya:yb, xa:xb]
        crop_ground = ground_axis[xa:xb] if ground_axis is not None else None
        local_nadir = int(nadir_x - xa)
        local_fig = surface_figure(
            crop_gray,
            crop_z,
            local_nadir,
            ground_axis_m=crop_ground,
            along_track_m_per_px=(float(along_res) if along_res > 0 else None),
            title=f"{det.id} • {det.class_name}",
            max_dim=260,
        )
        st.plotly_chart(local_fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

with tab_report:
    rows = []
    shadow_by_id = {e.detection_id: e for e in shadow_estimates}
    for det in detections:
        est = shadow_by_id.get(det.id)
        along_m, cross_m = metric_dimensions(det, ground_axis, float(along_res) if along_res > 0 else None)
        cx, cy = det.center
        signed_range = float(np.interp(cx, np.arange(W), ground_axis)) if ground_axis is not None else None
        out_lat = out_lon = None
        if enable_geo and signed_range is not None:
            out_lat, out_lon = geotag_detection(float(lat), float(lon), float(heading), signed_range)
        rows.append({
            "id": det.id,
            "class": det.class_name,
            "confidence_pct": round(det.confidence * 100, 2),
            "side": "starboard" if cx >= nadir_x else "port",
            "cross_track_range_m": round(abs(signed_range), 2) if signed_range is not None else None,
            "along_track_extent_m": round(along_m, 2) if along_m is not None else None,
            "cross_track_extent_m": round(cross_m, 2) if cross_m is not None else None,
            "shadow_height_m": round(est.height_m, 2) if est and est.shadow_found and est.height_m is not None else None,
            "shadow_quality_pct": round(est.quality * 100, 1) if est else None,
            "latitude": round(out_lat, 7) if out_lat is not None else None,
            "longitude": round(out_lon, 7) if out_lon is not None else None,
            "bbox_px": [round(det.x1,1), round(det.y1,1), round(det.x2,1), round(det.y2,1)],
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    report = {
        "source_image": uploaded.name,
        "image_size_px": {"width": W, "height": H},
        "model": model_label,
        "classes": ["shipwreck", "pipeline_cable", "ghost_net", "submarine_pipeline"],
        "geometry": {
            "nadir_x_px": int(nadir_x),
            "nadir_detection_confidence": round(float(nadir_conf), 4),
            "range_mode": range_mode_key,
            "range_at_edge_m": float(max_range),
            "sonar_altitude_m": float(altitude),
            "along_track_m_per_px": float(along_res) if along_res > 0 else None,
        },
        "contacts": rows,
        "disclaimer": "SSS backscatter is not bathymetry. shadow_height_m is an automatic physics-assisted estimate and must be validated before operational use.",
    }

    b1, b2 = st.columns(2)
    with b1:
        st.download_button(
            "Download JSON",
            data=json.dumps(report, indent=2),
            file_name=f"{Path(uploaded.name).stem}_contacts.json",
            mime="application/json",
            use_container_width=True,
        )
    with b2:
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{Path(uploaded.name).stem}_contacts.csv",
            mime="text/csv",
            use_container_width=True,
        )

with tab_method:
    st.markdown("### What this prototype is doing")
    st.markdown(
        """
1. **Detector** — runs the supplied YOLO26m checkpoint on the SSS image and returns the four trained classes.
2. **Nadir estimate** — searches the central region for the strongest persistent dark column instead of assuming the image centre is always the sonar track.
3. **Range geometry** — converts image columns to ground range. Raw slant-range imagery uses `sqrt(r² − H²)`; already corrected imagery is mapped directly in ground range.
4. **Shadow pairing** — for each detected contact, searches *away from nadir* for a stable dark acoustic-shadow run.
5. **Height estimate** — when target range, shadow end and altitude are available, estimates height using `h = H·L / (x + L)`.
6. **3D surface** — raises positive structural returns only inside detector ROIs. Dark shadows and the water column are **not** interpreted as negative elevation.
7. **Report** — outputs classification, model confidence, side, metric extent where scale is known, shadow-derived height where available, and optional WGS84 contact position.
        """
    )
    st.markdown("### What still requires raw survey data for engineering-grade output")
    st.markdown(
        """
- XTF/JSF ping-level timing and exact sample/range calibration
- towfish/AUV navigation rather than uncorrected vessel GPS
- heading, roll, pitch, heave and sensor mounting offsets
- layback correction for towed systems
- sound velocity and local seabed slope
- MBES/bathymetry fusion for a measured terrain surface
- multi-pass views for better target geometry and validation
        """
    )
    st.caption("The app deliberately distinguishes 'visual 3D' from 'metric 3D' so the dashboard stays technically defensible.")
