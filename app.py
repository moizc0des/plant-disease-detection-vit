import streamlit as st
import torch
import timm
import gdown
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import os

# ---------------- CONFIG ----------------
MODEL_PATH = "model/vit_plantdoc.pth"
NUM_CLASSES = 28
DISEASE_THRESHOLD = 0.35

CLASSES = [
    "Apple Scab Leaf", "Apple leaf", "Apple rust leaf", "Bell_pepper leaf",
    "Bell_pepper leaf spot", "Blueberry leaf", "Cherry leaf",
    "Corn Gray leaf spot", "Corn leaf blight", "Corn rust leaf",
    "Peach leaf", "Potato leaf early blight", "Potato leaf late blight",
    "Raspberry leaf", "Soyabean leaf", "Squash Powdery mildew leaf",
    "Strawberry leaf", "Tomato Early blight leaf", "Tomato Septoria leaf spot",
    "Tomato leaf", "Tomato leaf bacterial spot", "Tomato leaf late blight",
    "Tomato leaf mosaic virus", "Tomato leaf yellow virus",
    "Tomato mold leaf", "grape leaf", "grape leaf black rot",
    "Tomato two spotted spider mites leaf"
]

# ---------------- UI ----------------
st.set_page_config(page_title="Plant Disease Detection", layout="centered")
st.title("🌿 Plant Disease Detection using Vision Transformer")

# ---------------- HELPERS ----------------


def parse_class(label: str):
    parts = label.split(" leaf")
    plant = parts[0].strip()
    if len(parts) == 1 or parts[1].strip() == "":
        disease = "Healthy"
    else:
        disease = parts[1].replace("_", " ").strip()
    return plant, disease


def disease_probabilities(probs, classes):
    healthy_idxs = [
        i for i, c in enumerate(classes)
        if c.lower().endswith("leaf") and "blight" not in c.lower()
    ]
    healthy_prob = probs[healthy_idxs].sum().item()
    disease_prob = 1.0 - healthy_prob
    return disease_prob, healthy_prob


def risk_level(disease_prob):
    if disease_prob < 0.3:
        return "Low 🟢"
    elif disease_prob < 0.6:
        return "Medium 🟡"
    else:
        return "High 🔴"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=NUM_CLASSES
    )
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu"),
        strict=False
    )
    model.eval()
    return model


def generate_attention_map(model, image_tensor):
    image_tensor.requires_grad = True
    outputs = model(image_tensor)
    pred = outputs.argmax(dim=1)
    outputs[0, pred].backward()

    attn = image_tensor.grad.abs().mean(dim=1)[0]
    attn = attn / attn.max()

    attn = attn.numpy()
    attn = cv2.GaussianBlur(attn, (11, 11), 0)
    attn = np.clip(attn ** 1.5, 0, 1)
    return attn


def overlay_heatmap(img_pil, attn, scale=1.0):
    img = np.array(img_pil.resize((224, 224)))

    attn = np.clip(attn * scale, 0, 1)
    heatmap = cv2.applyColorMap(
        np.uint8(255 * attn),
        cv2.COLORMAP_JET
    )

    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)


# ---------------- LOAD MODEL ----------------
model = load_model()
if model is None:
    st.warning("⚠️ Model weights not available.")
    st.info("Run locally to enable predictions.")
    st.stop()

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ---------------- APP LOGIC ----------------
uploaded = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "png", "jpeg"]
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Leaf Image", use_container_width=True)

    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]

    pred_idx = torch.argmax(probs).item()
    plant, raw_disease = parse_class(CLASSES[pred_idx])
    disease_prob, healthy_prob = disease_probabilities(probs, CLASSES)
    risk = risk_level(disease_prob)

    if disease_prob > DISEASE_THRESHOLD:
        disease = raw_disease if raw_disease != "Healthy" else "Possible disease"
    else:
        disease = "Healthy"

    # ---------- DISPLAY ----------
    st.subheader("🧬 Diagnosis Result")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("🌱 Plant", plant)
        if disease == "Healthy":
            st.metric("🦠 Disease", "No disease detected")
        else:
            st.metric("🦠 Disease", disease)

    with col2:
        st.metric("Disease Probability", f"{disease_prob*100:.2f}%")
        st.metric("Healthy Probability", f"{healthy_prob*100:.2f}%")
        st.metric("Risk Level", risk)

    # ---------- TOP-3 ----------
    st.subheader("🔍 Possible Conditions")
    topk = torch.topk(probs, k=3)
    for i in range(3):
        st.write(f"- {CLASSES[topk.indices[i]]}: {topk.values[i]*100:.2f}%")

    # ---------- HOTSPOTS (ALWAYS SHOWN) ----------
    attn = generate_attention_map(model, x)

    if disease_prob > DISEASE_THRESHOLD:
        overlay = overlay_heatmap(img, attn, scale=1.0)
        st.subheader("🔥 Disease Hotspot Map")
    else:
        overlay = overlay_heatmap(img, attn, scale=0.6)
        st.subheader("🟡 Probable Disease Hotspot Map")

    st.image(
        overlay,
        caption="Attention-based regions of interest",
        use_container_width=True
    )
