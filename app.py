import streamlit as st
import cv2
import numpy as np
from keras.models import load_model


# Load trained emotion model
model = load_model("emotion_model.h5")

emotion_labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

# Load face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

st.title("AI Classroom Emotion & Attention Monitor")

st.write("Detect student emotions and classroom attention levels")

# Choose input method
option = st.radio("Choose Input Method", ["Upload Image", "Use Webcam"])

image = None

# IMAGE UPLOAD
if option == "Upload Image":

    uploaded_file = st.file_uploader("Upload Classroom Image", type=["jpg","jpeg","png"])

    if uploaded_file is not None:

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

# WEBCAM
if option == "Use Webcam":

    camera_image = st.camera_input("Capture Classroom Image")

    if camera_image is not None:

        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

# PROCESS IMAGE
if image is not None:

    st.info("Analyzing classroom engagement...")

    # Resize image to prevent memory error
    image = cv2.resize(image, (640,480))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30,30)
    )

    total_students = len(faces)

    attentive = 0
    bored = 0
    distracted = 0

    attention_map = {
        "Happy": "Attentive",
        "Neutral": "Attentive",
        "Surprise": "Attentive",
        "Sad": "Bored",
        "Angry": "Distracted",
        "Fear": "Distracted",
        "Disgust": "Distracted"
    }

    # Loop through detected faces
    for (x,y,w,h) in faces:

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face,(48,48))
        face = face / 255.0
        face = np.reshape(face,(1,48,48,1))

        prediction = model.predict(face)

        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]
        confidence = prediction[0][emotion_index] * 100

        attention_status = attention_map[emotion]

        if attention_status == "Attentive":
            attentive += 1
        elif attention_status == "Bored":
            bored += 1
        else:
            distracted += 1

        label = f"{emotion} ({confidence:.1f}%)"

        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(image,label,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    st.image(image, channels="BGR")

    # CLASSROOM STATISTICS
    if total_students > 0:

        attentive_percent = (attentive/total_students)*100
        bored_percent = (bored/total_students)*100
        distracted_percent = (distracted/total_students)*100

        st.subheader("Classroom Attention Statistics")

        st.write(f"Total Students Detected: {total_students}")

        st.success(f"Attentive Students: {attentive} ({attentive_percent:.1f}%)")

        st.warning(f"Bored Students: {bored} ({bored_percent:.1f}%)")

        st.error(f"Distracted Students: {distracted} ({distracted_percent:.1f}%)")

    else:

        st.warning("No students detected in the image")