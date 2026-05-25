# app.py
# Smart Exam Monitoring System using Streamlit + OpenCV

import streamlit as st
import cv2
import numpy as np
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

st.set_page_config(page_title="Smart Exam Monitoring System", layout="wide")

st.title("🎓 Smart Exam Monitoring System")
st.write("AI-based monitoring system for online examinations")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Session state
if "violations" not in st.session_state:
    st.session_state.violations = []

# Sidebar
st.sidebar.header("⚙️ Settings")
max_faces = st.sidebar.slider("Allowed Faces", 1, 3, 1)

# Video Transformer
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Violation detection
        if len(faces) > max_faces:
            violation = f"⚠️ Multiple faces detected at {datetime.now().strftime('%H:%M:%S')}"
            st.session_state.violations.append(violation)

            cv2.putText(
                img,
                "ALERT: MULTIPLE FACES!",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        # Status text
        cv2.putText(
            img,
            f"Faces Detected: {len(faces)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        return img

# Start webcam monitoring
st.subheader("📷 Live Exam Monitoring")

webrtc_streamer(
    key="exam-monitor",
    video_transformer_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Violations Section
st.subheader("🚨 Violations Log")

if st.session_state.violations:
    for v in st.session_state.violations:
        st.error(v)
else:
    st.success("No violations detected.")

# Footer
st.markdown("---")
st.caption("Smart Exam Monitoring System using Streamlit + OpenCV")
