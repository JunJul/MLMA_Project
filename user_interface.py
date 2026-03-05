import streamlit as st
from PIL import Image
import numpy as np
import random

import torch
import torchvision.transforms as transforms
import yaml
import re
from pathlib import Path
from importlib import import_module

from utils import gradcam_overlay


CLASS_NAMES = ["COVID19", "NORMAL", "PNEUMONIA"]

# Must be the first Streamlit command
st.set_page_config(page_title="Chest X-ray Demo", layout="wide")

st.title("🫁 Chest X-ray Classification Demo")
st.caption(
    "Upload a chest X-ray image → predicted label, confidence (ensemble), uncertainty warning, "
    "and an attention heatmap (if a checkpoint is available)."
)

# ----------------------------
# Sidebar controls (keep original)
# ----------------------------
st.sidebar.header("Controls")
# One slider to control ALL image sizes
img_w = st.sidebar.slider("Image width", 200, 900, 360, 10)

uncertainty_threshold = st.sidebar.slider("Uncertainty threshold", 0.00, 0.30, 0.10, 0.01)
seed = st.sidebar.number_input("Mock seed (for reproducibility)", value=0, step=1)
force_warning = st.sidebar.checkbox("Force high-uncertainty (for demo screenshot)", value=False)

# Optional Grad-CAM (does not affect mock prediction logic)
st.sidebar.markdown("---")
st.sidebar.header("Heatmap (optional)")
enable_heatmap = st.sidebar.checkbox("Enable heatmap (requires checkpoint)", value=False)
config_path = st.sidebar.selectbox(
    "Model config (for heatmap)",
    ["model_confs/ResNet50.yaml", "model_confs/ResNetSE.yaml", "model_confs/ResNetCBAM.yaml"],
    index=1
)


@st.cache_resource
def load_model_for_heatmap(cfg_path: str):
    """
    Loads model + checkpoint for heatmap only.
    This does NOT change the mock prediction / uncertainty logic.
    """
    cfg_file = Path(cfg_path)
    if not cfg_file.exists():
        return None, None, f"Config not found: {cfg_path}"

    cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))

    # Build model (match pipeline.py style)
    model_type = cfg["model"]["type"]  # e.g., "models.ResNetSE"
    params = cfg["model"].get("params", {}) or {}
    module = import_module(model_type)
    class_name = model_type.split(".")[-1]
    if not hasattr(module, class_name):
        return None, None, f"Model class '{class_name}' not found in module '{model_type}'."

    model_cls = getattr(module, class_name)
    model = model_cls(**params)

    # Locate experiment directory (match pipeline.py: output_dir / f"{model_name}_{loss_name}")
    output_dir = Path(cfg.get("output_dir", "experiments"))
    loss_type = cfg.get("loss", {}).get("type", "unknown")
    loss_name = loss_type.split(".")[-1]

    exp_dir = output_dir / f"{class_name}_{loss_name}"
    models_dir = exp_dir / "models"
    exp_cfg_path = exp_dir / "config.yaml"

    best_epoch = None
    if exp_cfg_path.exists():
        exp_cfg = yaml.safe_load(exp_cfg_path.read_text(encoding="utf-8"))
        best_epoch = exp_cfg.get("best_epoch", None)

    ckpt_path = None

    # 1) Try best_epoch checkpoint
    if best_epoch is not None:
        p = models_dir / f"{class_name}_epoch_{best_epoch}.pt"
        if p.exists():
            ckpt_path = p

    # 2) Fallback: choose largest epoch checkpoint
    if ckpt_path is None and models_dir.exists():
        cand = list(models_dir.glob(f"{class_name}_epoch_*.pt"))
        if cand:
            def get_ep(pp: Path):
                m = re.search(r"_epoch_(\d+)\.pt$", pp.name)
                return int(m.group(1)) if m else -1
            cand.sort(key=get_ep, reverse=True)
            ckpt_path = cand[0]

    if ckpt_path is None:
        return None, None, f"Checkpoint not found under: {models_dir}"

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    return model, device, f"Loaded checkpoint: {ckpt_path}"


# ----------------------------
# Upload
# ----------------------------
uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Please upload an X-ray image to start.")
    st.stop()

img = Image.open(uploaded).convert("RGB")

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input image")
    #  Make input image smaller
    st.image(img, width=img_w)

# ----------------------------
# Mock inference (keep original logic)
# ----------------------------
random.seed(seed)
np.random.seed(seed)

raw = np.random.rand(len(CLASS_NAMES))
probs = raw / raw.sum()

pred_idx = int(np.argmax(probs))
pred_label = CLASS_NAMES[pred_idx]
confidence = float(probs[pred_idx])

uncertainty = float(np.random.uniform(0.01, 0.20))
if force_warning:
    uncertainty = max(uncertainty, uncertainty_threshold + 0.05)

# Results panel
with col2:
    st.subheader("Prediction")

    st.metric("Predicted Label", pred_label)

    st.markdown("**Confidence (Ensemble)**")
    st.progress(min(confidence, 1.0))
    st.write(f"{confidence:.3f}")

    st.markdown("**Uncertainty**")
    st.write(f"{uncertainty:.3f}")

    if uncertainty >= uncertainty_threshold:
        st.error(" High uncertainty detected. Specialist review recommended.")
    else:
        st.success("Low uncertainty. Model is relatively confident.")

    st.markdown("---")
    st.subheader("Class probabilities")

    for i, name in enumerate(CLASS_NAMES):
        prob = float(probs[i])
        c1, c2 = st.columns([3, 1])
        with c1:
            st.progress(min(prob, 1.0))
        with c2:
            st.write(f"{name}: {prob:.3f}")

# ----------------------------
# Explanation / Heatmap section
# ----------------------------
st.markdown("---")
st.subheader("Explanation")
st.caption("Heatmap overlay showing where the model focuses in the image (if enabled and checkpoint exists).")

if not enable_heatmap:
    #  Make placeholder image smaller too
    st.image(img, caption="Heatmap overlay is disabled (showing input image).", width=img_w)
    st.info("Enable the heatmap in the sidebar to generate a Grad-CAM overlay (requires a trained checkpoint).")
else:
    model, device, status = load_model_for_heatmap(config_path)
    st.write(status)

    if model is None:
        st.image(img, caption="Heatmap unavailable (showing input image).", width=img_w)
        st.warning("Heatmap cannot be generated because the model checkpoint was not found.")
    else:
        eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        x = eval_transform(img).unsqueeze(0).to(device)

        try:
            cam_img, probs_cam, pred_cam = gradcam_overlay(model, x)
            # Make heatmap smaller too
            st.image(cam_img, caption=f"Heatmap overlay (pred: {CLASS_NAMES[int(pred_cam)]})", width=img_w)
        except Exception as e:
            st.image(img, caption="Heatmap error (showing input image).", width=img_w)
            st.error(f"Failed to generate heatmap: {e}")

st.markdown("---")
st.caption(
    "Prototype UI: prediction/uncertainty are mock outputs. "
    "Heatmap can be real if you have trained checkpoints (.pt)."
)