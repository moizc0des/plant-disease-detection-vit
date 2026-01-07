import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import os

MODEL_PATH = "model/vit_plantdoc.pth"

st.set_page_config(page_title="Plant Disease Detection", layout="centered")
st.title("🌿 Plant Disease Detection using Vision Transformer")


@st.cache_resource
def load_model():
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=28
    )
    model.load_state_dict(torch.load(
        MODEL_PATH, map_location="cpu"), strict=False)
    model.eval()
    return model


model = load_model()
classes = sorted(os.listdir("data/train"))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Leaf Image", use_column_width=True)

    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]

    pred_idx = torch.argmax(probs).item()
    confidence = probs[pred_idx].item()

    st.subheader("Prediction")
    st.success(classes[pred_idx])
    st.progress(float(confidence))
    st.caption(f"Confidence: {confidence*100:.2f}%")
