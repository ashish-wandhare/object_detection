import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont, ImageOps
import pandas as pd

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(
    page_title="Smart Store Product Detection",
    layout="wide"
)

# -------------------------------
# Load Model (cached)
# -------------------------------

@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt")
        model.to("cpu")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()


model = load_model()
# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.title("Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=1.0,
    value=0.25,
    step=0.05
)

# -------------------------------
# App Title
# -------------------------------
st.title("🛒 Smart Store Product Detection System")
st.markdown("Upload an image to detect products from cake, candy, cereal, chips, chocolate, coffee, fish, honey, jam, milk, oil, pasta, rice, soda, sugar, tea, vinegar, water categories" ")
# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Helper: Draw Bounding Boxes
# -------------------------------
def draw_boxes(image, results, model):
    draw = ImageDraw.Draw(image)

    box_color = (0, 0, 255)
    text_color = (255, 255, 255)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

        # Label text
        text = f"{label} {conf:.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        draw.rectangle(
            [x1, y1 - th - 4, x1 + tw + 4, y1],
            fill=box_color
        )
        draw.text(
            (x1 + 2, y1 - th - 2),
            text,
            fill=text_color,
            font=font
        )

        detections.append({
            "Class": label,
            "Confidence": round(conf, 3),
            "X1": int(x1),
            "Y1": int(y1),
            "X2": int(x2),
            "Y2": int(y2)
        })

    return image, detections

# -------------------------------
# Run Inference
# -------------------------------
if uploaded_file:
    col1, col2 = st.columns(2)

    image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
    MAX_SIZE = 1280
    image.thumbnail((MAX_SIZE, MAX_SIZE))
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with st.spinner("Running detection..."):
        results = model.predict(image, conf=conf_threshold)[0]
        annotated_image, detections = draw_boxes(image.copy(), results, model)

    with col2:
        st.subheader("Detected Image")
        st.image(annotated_image, use_container_width=True)

    # -------------------------------
    # Detection Results Table
    # -------------------------------
    st.subheader("Detection Results")

    if detections:
        df = pd.DataFrame(detections)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No objects detected.")
