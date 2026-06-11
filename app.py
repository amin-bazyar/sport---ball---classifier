import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet

CLASS_NAMES = [
    'american_football', 'baseball', 'basketball', 'billiard_ball',
    'bowling_ball', 'cricket_ball', 'football', 'golf_ball',
    'hockey_puck', 'rugby_ball', 'shuttlecock', 'table_tennis_ball',
    'tennis_ball', 'volleyball', 'frisbee'
]

@st.cache_resource
def load_model():
    model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=15)
    model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

st.title('Sports Ball Classifier')
st.write('Upload an image to identify the type of sports ball.')

uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    model = load_model()
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top5 = torch.topk(probs, 5)
    st.subheader('Results:')
    for i in range(5):
        idx = top5.indices[i].item()
        prob = top5.values[i].item()
        st.write(f'{CLASS_NAMES[idx]}: {prob*100:.1f}%')
