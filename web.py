import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile
import numpy as np
from tensorflow.keras.models import model_from_json


logo_detection_model_path = "model/angelhack_yolov9.pt"

face_classifier = cv2.CascadeClassifier(r"./model/haarcascade_frontalface_default.xml")
model_json_file = "./model/model.json"
model_weights_file = "./model/Latest_Model.h5"

st.set_page_config(
    page_title="Object Detection using YOLOv9",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.header("Video Detection")

    option = st.selectbox(
        "How would you like to be contacted?",
        ("Logo detect", "Emotion detect", "Drink"),
        index=None,
        placeholder="Select contact method...",
    )
    source_vid = st.sidebar.file_uploader("Choose a file", type=["jpeg", "jpg", "png", "webp"])

    # confidence = float(st.slider(
    #     "Select Model Confidence", 25, 100, 40)) / 100

if option == "Logo detect" and source_vid is not None:
    try:
        model = YOLO(logo_detection_model_path)
    except Exception as ex:
        st.error(
            f"Unable to load model. Check the specified path: {logo_detection_model_path}")
        st.error(ex)
    st.write("Model loaded successfully!")

    if st.sidebar.button('Logo Detect'):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(source_vid.read())
            temp_filename = temp_file.name

            vid_cap = cv2.VideoCapture(temp_filename)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    image = cv2.resize(image, (720, int(720 * (9 / 16))))
                    res = model.predict(image, conf=0.1)
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

elif option == "Emotion detect" and source_vid is not None:

    with open(model_json_file, "r") as json_file:
        loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)
        classifier.load_weights(model_weights_file)

    if st.sidebar.button('Emotion Detect'):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(source_vid.read())
            temp_filename = temp_file.name

            image = cv2.imread(temp_filename)
            if image is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_classifier.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    fc = gray[y:y+h, x:x+w]
                    roi = cv2.resize(fc, (48, 48))
                    roi = roi[np.newaxis, :, :, np.newaxis]  # Add batch and channel dimensions
                    pred = classifier.predict(roi)
                    text_idx = np.argmax(pred)
                    text_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
                    text = text_list[text_idx]

                    cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

                st.image(image, caption='Detected Emotions', channels="BGR", use_column_width=True)
            else:
                st.error("Error: Unable to read the uploaded image.")
