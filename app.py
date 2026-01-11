import streamlit as st
import torch
import timm
import gdown
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import os
import tempfile
from gtts import gTTS
from googletrans import Translator
from twilio.rest import Client

# ---------------- CONFIG ----------------
NUM_CLASSES = 28
DISEASE_THRESHOLD = 0.35

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "vit_plantdoc.pth")
GDRIVE_FILE_ID = "172eRtmX0zQKaw_O_WePvoQTCr_qG-Z8G"

LANG_MAP = {"English": "en", "Hindi": "hi", "Telugu": "te"}

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

DISEASE_ADVICE = {
    "Tomato Early blight": {
        "cause": "Fungal infection",
        "treatment": "Spray Mancozeb or Chlorothalonil once every 7 days",
        "prevention": "Avoid overhead irrigation and remove infected leaves"
    },
    "Potato leaf late blight": {
        "cause": "Fungal infection",
        "treatment": "Use Copper fungicide or Metalaxyl",
        "prevention": "Ensure proper drainage and crop rotation"
    }
}

# ---------------- UI ----------------
st.set_page_config(page_title="Plant Disease Detection", layout="centered")
st.title("🌿 Smart Plant Disease Detection System")

# ---------------- HELPERS ----------------


def parse_class(label):
    parts = label.split(" leaf")
    plant = parts[0].strip()
    disease = "Healthy" if len(
        parts) == 1 else parts[1].replace("_", " ").strip()
    return plant, disease


def disease_probabilities(probs):
    healthy_idxs = [
        i for i, c in enumerate(CLASSES)
        if c.lower().endswith("leaf") and "blight" not in c.lower()
    ]
    healthy_prob = probs[healthy_idxs].sum().item()
    return 1 - healthy_prob, healthy_prob


def risk_level(p):
    if p < 0.3:
        return "Low 🟢"
    elif p < 0.6:
        return "Medium 🟡"
    else:
        return "High 🔴"


@st.cache_resource
def load_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model (first run only)..."):
            gdown.download(
                f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
                MODEL_PATH,
                quiet=False
            )
    model = timm.create_model("vit_base_patch16_224",
                              pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(
        MODEL_PATH, map_location="cpu"), strict=False)
    model.eval()
    return model


def generate_attention(model, x):
    x.requires_grad = True
    out = model(x)
    out[0, out.argmax()].backward()
    attn = x.grad.abs().mean(dim=1)[0]
    attn = (attn / attn.max()).numpy()
    attn = cv2.GaussianBlur(attn, (11, 11), 0)
    return np.clip(attn ** 1.5, 0, 1)


def overlay_heatmap(img, attn, scale=1.0):
    img = np.array(img.resize((224, 224)))
    heat = cv2.applyColorMap(np.uint8(255 * attn * scale), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.6, heat, 0.4, 0)


def generate_audio(text, lang):
    tts = gTTS(text=text, lang=lang)
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp.name)
    return temp.name


def send_sms(phone, text):
    client = Client(
        st.secrets["TWILIO_SID"],
        st.secrets["TWILIO_AUTH"]
    )
    client.messages.create(
        body=text,
        from_=st.secrets["TWILIO_PHONE"],
        to=phone
    )


# ---------------- LOAD MODEL ----------------
model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- APP ----------------
uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])
language = st.selectbox("Select Language", list(LANG_MAP.keys()))
phone = st.text_input("Phone number for SMS (optional)",
                      placeholder="+91XXXXXXXXXX")

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Leaf Image", use_container_width=True)

    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x)[0], dim=0)

    idx = torch.argmax(probs).item()
    plant, raw_disease = parse_class(CLASSES[idx])
    disease_prob, healthy_prob = disease_probabilities(probs)
    risk = risk_level(disease_prob)

    disease = raw_disease if disease_prob > DISEASE_THRESHOLD else "Healthy"

    st.subheader("🧬 Diagnosis Result")
    st.metric("🌱 Plant", plant)
    st.metric("🦠 Disease", "No disease detected" if disease ==
              "Healthy" else disease)
    st.metric("Risk Level", risk)

    advice = DISEASE_ADVICE.get(disease, None)
    if advice:
        st.subheader("🧑‍🌾 Farmer Recommendation")
        st.write(f"**Cause:** {advice['cause']}")
        st.write(f"**Treatment:** {advice['treatment']}")
        st.write(f"**Prevention:** {advice['prevention']}")

    text = (
        f"This is a {plant} leaf. "
        f"Disease detected: {disease}. "
        f"Risk level is {risk}. "
    )
    if advice:
        text += f"Recommended treatment is {advice['treatment']}."

    translator = Translator()
    translated = translator.translate(text, dest=LANG_MAP[language]).text
    audio = generate_audio(translated, LANG_MAP[language])
    st.audio(audio)

    if phone and st.button("📩 Send SMS"):
        send_sms(phone, translated)
        st.success("SMS sent successfully!")

    attn = generate_attention(model, x)
    overlay = overlay_heatmap(img, attn, 1.0 if disease != "Healthy" else 0.6)
    st.subheader("🔥 Disease Hotspot Map" if disease !=
                 "Healthy" else "🟡 Probable Disease Hotspot Map")
    st.image(overlay, use_container_width=True)
