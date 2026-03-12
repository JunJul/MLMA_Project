import streamlit as st
from PIL import Image
import numpy as np
import random

CLASS_NAMES = ["COVID19", "NORMAL", "PNEUMONIA"]

st.set_page_config(page_title="Chest X-ray Demo", layout="wide")
st.title("🫁 Chest X-ray Classification Demo")
st.caption("Upload a chest X-ray image → predicted label, confidence (ensemble), uncertainty warning, and model attention (Grad-CAM).")

# Sidebar controls
st.sidebar.header("Controls")
uncertainty_threshold = st.sidebar.slider("Uncertainty threshold", 0.00, 0.30, 0.10, 0.01)
seed = st.sidebar.number_input("Mock seed (for reproducibility)", value=0, step=1)
force_warning = st.sidebar.checkbox("Force high-uncertainty (for demo screenshot)", value=False)

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Please upload an X-ray image to start.")
    st.stop()

img = Image.open(uploaded).convert("RGB")

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input image")
    st.image(img, use_container_width=True)

# Mock inference (replace later with real ensemble inference)
random.seed(seed)
np.random.seed(seed)

# Mock class probabilities (sum to 1)
raw = np.random.rand(len(CLASS_NAMES))
probs = raw / raw.sum()

pred_idx = int(np.argmax(probs))
pred_label = CLASS_NAMES[pred_idx]
confidence = float(probs[pred_idx])

# Mock uncertainty (future: std across multiple models)
uncertainty = float(np.random.uniform(0.01, 0.20))
if force_warning:
    uncertainty = max(uncertainty, uncertainty_threshold + 0.05)

# Results panel
with col2:
    st.subheader("Prediction")

    # 1) Predicted Label 
    st.metric("Predicted Label", pred_label)

    # 3) Confidence Score (Ensemble Model) 
    st.markdown("**Confidence (Ensemble)**")
    st.progress(min(confidence, 1.0))
    st.write(f"{confidence:.3f}")

    # 4) Uncertainty Warning 
    st.markdown("**Uncertainty**")
    st.write(f"{uncertainty:.3f}")

    if uncertainty >= uncertainty_threshold:
        st.error("⚠️ High uncertainty detected. Specialist review recommended.")
    else:
        st.success("✅ Low uncertainty. Model is relatively confident.")

    st.markdown("---")
    st.subheader("Class probabilities")

    for i, name in enumerate(CLASS_NAMES):
        prob = float(probs[i])

        col1, col2 = st.columns([3,1])
    
        with col1:
            st.progress(prob)
    
        with col2:
            st.write(f"{name}: {prob:.3f}")

# 2) Present how the model is looking into the image  (Grad-CAM section)
st.markdown("---")
st.subheader("Explanation (Grad-CAM)")
st.caption("This section shows where the model focuses in the image. (Placeholder until checkpoints are available.)")


st.image(img, caption="Grad-CAM heatmap will be displayed here.", use_container_width=True)

st.info("Grad-CAM is not connected yet. Once model checkpoints are available, this area will show a heatmap overlay indicating model attention.")

# Footer note
st.markdown("---")
st.caption("Prototype UI: mock outputs now; will be replaced with real ensemble inference + Grad-CAM once training artifacts (.pt) are ready.")