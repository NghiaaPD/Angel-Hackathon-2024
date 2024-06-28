import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile

model_path = "model/angelhack_yolov9.pt"

st.set_page_config(
    page_title="Object Detection using YOLOv9",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.header("Video Detection")
    source_vid = st.sidebar.file_uploader("Choose a file", type=["mp4", "mov", "avi", "mkv"])

    # Model Options
    confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 40)) / 100

st.title("Object Detection using YOLOv9")

try:
    model = YOLO(model_path).cuda()
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)
st.write("Model loaded successfully!")

if source_vid is not None:
    if st.sidebar.button('Detect Objects'):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(source_vid.read())
            temp_filename = temp_file.name

        vid_cap = cv2.VideoCapture(temp_filename)
        st_frame = st.empty()
        while vid_cap.isOpened():
            success, image = vid_cap.read()
            if success:
                image = cv2.resize(image, (720, int(720 * (9 / 16))))
                res = model.predict(image, conf=confidence)
                result_tensor = res[0].boxes
                res_plotted = res[0].plot()
                st_frame.image(res_plotted,
                               caption='Detected Video',
                               channels="BGR",
                               use_column_width=True
                               )
            else:
                vid_cap.release()
                break
