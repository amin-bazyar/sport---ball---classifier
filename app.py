import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from PIL import Image
import gdown
import os

# Classes
CLASS_NAMES = [
    'american_football', 'baseball', 'basketball', 'billiard_ball',
    'bowling_ball', 'cricket_ball', 'football', 'golf_ball',
    'hockey_puck', 'rugby_ball', 'shuttlecock', 'table_tennis_ball',
    'tennis_ball', 'volleyball', 'frisbee'
]

MODEL_PATH = 'best_model.pth'
FILE_ID = '1FgJz4uGtusj_2bazxP_hs_BjQ1o9vdup'

# Load model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)

    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 15)

    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(state_dict,strict=False)

    model.eval()
    return model

# Image transform
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# UI
st.set_page_config(page_title="Sports Ball Classifier", layout="wide")

st.title("Sports Ball Classifier")
st.write("Upload an image to identify the type of sports ball")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Input Image", use_column_width=True)

    model = load_model()
    img_tensor = transform(image).unsqueeze(0)

    with st.spinner("Analyzing image..."):
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]

    top5 = torch.topk(probs, 5)

    top1_idx = top5.indices[0].item()
    top1_prob = top5.values[0].item()

    with col2:
        st.subheader("Prediction")
        st.markdown(f"## {CLASS_NAMES[top1_idx]}")
        st.write(f"Confidence: {top1_prob*100:.2f}%")

        if top1_prob > 0.9:
            st.success("High confidence")
        elif top1_prob > 0.7:
            st.warning("Medium confidence")
        else:
            st.error("Low confidence")

    st.subheader("Top Predictions")

    for i in range(5):
        idx = top5.indices[i].item()
        prob = top5.values[i].item()

        st.write(CLASS_NAMES[idx])
        st.progress(float(prob))
