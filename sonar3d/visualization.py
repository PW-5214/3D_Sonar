from __future__ import annotations

import cv2
import numpy as np
import plotly.graph_objects as go

from .geometry import normalize_percentile
from .inference import Detection


SONAR_SCALE = [
    [0.00, "rgb(5,4,2)"],
    [0.08, "rgb(25,15,3)"],
    [0.20, "rgb(66,35,2)"],
    [0.36, "rgb(110,60,2)"],
    [0.55, "rgb(162,95,3)"],
    [0.72, "rgb(212,137,10)"],
    [0.88, "rgb(242,180,35)"],
    [1.00, "rgb(255,225,105)"],
]


def _resize_grid(arr: np.ndarray, max_dim: int = 300, interpolation=cv2.INTER_AREA):
    h, w = arr.shape
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return arr, 1.0
    out = cv2.resize(arr, (max(2, int(w * scale)), max(2, int(h * scale))), interpolation=interpolation)
    return out, scale


def surface_figure(
    gray: np.ndarray,
    z: np.ndarray,
    nadir_x: int,
    ground_axis_m: np.ndarray | None = None,
    along_track_m_per_px: float | None = None,
    title: str = "Detection-aware 3D sonar scene",
    max_dim: int = 300,
) -> go.Figure:
    color = normalize_percentile(gray)
    z_small, scale = _resize_grid(z.astype(np.float32), max_dim=max_dim, interpolation=cv2.INTER_AREA)
    color_small = cv2.resize(color.astype(np.float32), (z_small.shape[1], z_small.shape[0]), interpolation=cv2.INTER_AREA)

    Hs, Ws = z_small.shape
    if ground_axis_m is not None:
        x_orig = np.linspace(0, len(ground_axis_m) - 1, Ws)
        x = np.interp(x_orig, np.arange(len(ground_axis_m)), ground_axis_m)
        x_title = "Cross-track ground range (m)"
    else:
        x = (np.arange(Ws) / max(scale, 1e-9)) - nadir_x
        x_title = "Cross-track pixels from nadir"

    if along_track_m_per_px is not None and along_track_m_per_px > 0:
        y = np.arange(Hs) / max(scale, 1e-9) * float(along_track_m_per_px)
        y_title = "Along-track distance (m)"
    else:
        y = np.arange(Hs) / max(scale, 1e-9)
        y_title = "Along-track pixels"

    X, Y = np.meshgrid(x, y)
    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=z_small,
                surfacecolor=color_small,
                colorscale=SONAR_SCALE,
                cmin=0,
                cmax=1,
                showscale=False,
                hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>",
                lighting=dict(ambient=0.45, diffuse=0.85, roughness=0.86, specular=0.14, fresnel=0.05),
                lightposition=dict(x=-200, y=-300, z=500),
            )
        ]
    )
    fig.update_layout(
        title=title,
        paper_bgcolor="#081018",
        plot_bgcolor="#081018",
        font=dict(color="#dbe7ee"),
        margin=dict(l=0, r=0, b=0, t=45),
        scene=dict(
            bgcolor="#081018",
            xaxis=dict(title=x_title, gridcolor="#24323d", zerolinecolor="#3b4d58"),
            yaxis=dict(title=y_title, gridcolor="#24323d", zerolinecolor="#3b4d58", autorange="reversed"),
            zaxis=dict(title="Estimated relief", gridcolor="#24323d", zerolinecolor="#3b4d58"),
            camera=dict(eye=dict(x=1.45, y=-1.65, z=1.05)),
            aspectmode="manual",
            aspectratio=dict(x=2.0, y=max(0.7, Hs / max(Ws, 1) * 2.0), z=0.55),
        ),
    )
    return fig


def crop_detection(gray: np.ndarray, z: np.ndarray, det: Detection, pad: float = 0.45):
    h, w = gray.shape
    x1, y1, x2, y2 = det.bbox
    bw, bh = x2 - x1, y2 - y1
    xa = max(0, int(x1 - bw * pad))
    xb = min(w, int(x2 + bw * pad) + 1)
    ya = max(0, int(y1 - bh * pad))
    yb = min(h, int(y2 + bh * pad) + 1)
    return gray[ya:yb, xa:xb], z[ya:yb, xa:xb], (xa, ya, xb, yb)
