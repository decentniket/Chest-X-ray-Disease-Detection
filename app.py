import torch.nn as nn
import torchvision.models as models
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# ======================
# CONFIG
# ======================
MODEL_PATH = "model.pt"
IMG_SIZE = 256

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    # recreate model
    model = models.resnet18(pretrained=False)

    # change final layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 1)

    # load weights (state_dict)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

    model.eval()
    return model
model = load_model()

# ======================
# IMAGE TRANSFORM
# ======================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ======================
# UI
# ======================
st.set_page_config(page_title="Chest Disease Detection", layout="centered")

st.title("🩺 Chest X-ray Disease Detection")
st.write("Upload a chest X-ray image to detect Pneumonia")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    with st.spinner("Analyzing..."):
        with torch.no_grad():
            output = model(img)

            # handle binary or multi-class
            if output.shape[1] == 1:
                prob = torch.sigmoid(output).item()
            else:
                prob = torch.softmax(output, dim=1)[0][1].item()

    label = "PNEUMONIA" if prob > 0.5 else "NORMAL"

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {prob:.4f}")

    # colored output
    if label == "PNEUMONIA":
        st.error("⚠️ Pneumonia Detected")
    else:
        st.success("✅ Normal")