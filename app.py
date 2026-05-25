# app.py
# =========================================================
# SMART EXAM MONITORING SYSTEM — PROFESSIONAL UI/UX
# =========================================================

import streamlit as st
import cv2
import av
import numpy as np
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart Exam Monitoring",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a, #111827);
    color: white;
}

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main Title */
.main-title {
    font-size: 42px;
    font-weight: 700;
    color: white;
    margin-bottom: 5px;
}

.subtitle {
    color: #94a3b8;
    font-size: 18px;
    margin-bottom: 25px;
}

/* Glassmorphism */
.glass {
    background: rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}

/* Metric Cards */
.metric-card {
    background: rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 20px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 6px 25px rgba(0,0,0,0.25);
}

.metric-title {
    color: #94a3b8;
    font-size: 14px;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
    color: white;
}

/* Alert Box */
.alert-box {
    background: rgba(239,68,68,0.15);
    border-left: 5px solid #ef4444;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 10px;
    color: white;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg,#2563eb,#7c3aed);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 20px;
    font-weight: 600;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0b1120;
}

/* Webcam container */
video {
    border-radius: 20px !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================

if "violations" not in st.session_state:
    st.session_state.violations = []

if "faces" not in st.session_state:
    st.session_state.faces = 0

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🎓 Smart Exam Monitoring Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered online examination monitoring system</div>',
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("⚙️ Control Panel")

    st.markdown("---")

    monitoring = st.toggle("Enable Monitoring", value=True)

    allowed_faces = st.slider(
        "Allowed Faces",
        min_value=1,
        max_value=3,
        value=1
    )

    sensitivity = st.slider(
        "Detection Sensitivity",
        1,
        10,
        5
    )

    st.markdown("---")

    st.subheader("🛡️ Features")

    st.write("✅ Face Detection")
    st.write("✅ Live Monitoring")
    st.write("✅ Violation Alerts")
    st.write("✅ AI Surveillance")
    st.write("✅ Real-time Analytics")

    st.markdown("---")

    st.caption("Version 2.0")

# =========================================================
# METRICS
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Monitoring Status</div>
        <div class="metric-value">🟢 ACTIVE</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Violations</div>
        <div class="metric-value">{len(st.session_state.violations)}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Detected Faces</div>
        <div class="metric-value">{st.session_state.faces}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# =========================================================
# FACE DETECTION MODEL
# =========================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# =========================================================
# VIDEO PROCESSOR
# =========================================================

class VideoProcessor(VideoTransformerBase):

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=sensitivity,
            minSize=(30, 30)
        )

        st.session_state.faces = len(faces)

        # Face rectangles
        for (x, y, w, h) in faces:

            cv2.rectangle(
                img,
                (x, y),
                (x+w, y+h),
                (0, 255, 255),
                3
            )

            cv2.putText(
                img,
                "Candidate",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2
            )

        # Violation Detection
        if monitoring:

            if len(faces) > allowed_faces:

                violation = (
                    f"Multiple faces detected at "
                    f"{datetime.now().strftime('%H:%M:%S')}"
                )

                if violation not in st.session_state.violations:
                    st.session_state.violations.append(violation)

                cv2.putText(
                    img,
                    "ALERT : MULTIPLE FACES DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

        # Overlay Panel
        overlay = img.copy()

        cv2.rectangle(
            overlay,
            (0,0),
            (360,90),
            (15,23,42),
            -1
        )

        img = cv2.addWeighted(
            overlay,
            0.5,
            img,
            0.5,
            0
        )

        # Overlay Text
        cv2.putText(
            img,
            f"Faces : {len(faces)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255,255,255),
            2
        )

        cv2.putText(
            img,
            datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        return img

# =========================================================
# MAIN LAYOUT
# =========================================================

left, right = st.columns([2.2, 1])

# =========================================================
# LIVE CAMERA
# =========================================================

with left:

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("📷 Live AI Monitoring")

    webrtc_streamer(
        key="exam-monitor",
        video_transformer_factory=VideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# ALERTS PANEL
# =========================================================

with right:

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("🚨 Security Alerts")

    if st.session_state.violations:

        for alert in reversed(st.session_state.violations[-5:]):

            st.markdown(
                f'''
                <div class="alert-box">
                ⚠️ {alert}
                </div>
                ''',
                unsafe_allow_html=True
            )

    else:
        st.success("No suspicious activity detected")

    st.markdown("---")

    st.subheader("📊 Exam Integrity")

    score = max(
        100 - len(st.session_state.violations) * 10,
        0
    )

    st.progress(score)

    st.metric(
        label="Integrity Score",
        value=f"{score}%"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.write("")

st.markdown("""
<center>
<p style='color:#94a3b8;'>
Smart Exam Monitoring System • AI Proctoring Dashboard
</p>
</center>
""", unsafe_allow_html=True)
