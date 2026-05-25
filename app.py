# ============================================================
# NEXT-GEN SMART EXAM MONITORING SYSTEM
# Ultra Modern Streamlit UI + AI Monitoring
# ============================================================

import streamlit as st
import cv2
import av
import numpy as np
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Exam Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM UI
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Background */
.stApp {
    background:
        radial-gradient(circle at top left, #1e293b, #020617);
    color: white;
}

/* Hide Streamlit Branding */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Main Hero */
.hero {
    padding: 25px;
    border-radius: 30px;
    background: linear-gradient(135deg,#111827,#1e293b);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0px 10px 40px rgba(0,0,0,0.4);
    margin-bottom: 20px;
}

.hero-title {
    font-size: 52px;
    font-weight: 800;
    color: white;
}

.hero-sub {
    color: #94a3b8;
    font-size: 18px;
}

/* Glass Cards */
.glass {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 22px;
    backdrop-filter: blur(14px);
    box-shadow: 0px 10px 30px rgba(0,0,0,0.25);
}

/* Metrics */
.metric-card {
    background: linear-gradient(145deg,#0f172a,#111827);
    padding: 24px;
    border-radius: 22px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.06);
}

.metric-title {
    color: #94a3b8;
    font-size: 14px;
}

.metric-value {
    color: white;
    font-size: 34px;
    font-weight: 800;
}

/* Alerts */
.alert {
    background: rgba(239,68,68,0.15);
    border-left: 5px solid #ef4444;
    padding: 14px;
    border-radius: 14px;
    margin-bottom: 10px;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #020617;
}

/* Webcam */
video {
    border-radius: 24px !important;
    overflow: hidden;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg,#3b82f6,#8b5cf6);
    color: white;
    border-radius: 14px;
    border: none;
    padding: 12px 22px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATES
# ============================================================

if "alerts" not in st.session_state:
    st.session_state.alerts = []

if "face_count" not in st.session_state:
    st.session_state.face_count = 0

if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">🛡️ AI Exam Shield</div>
    <div class="hero-sub">
        Real-time AI-powered online examination surveillance platform
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ System Settings")

    monitoring = st.toggle("Enable AI Monitoring", True)

    allowed_faces = st.selectbox(
        "Allowed Candidates",
        [1,2,3],
        index=0
    )

    sensitivity = st.slider(
        "Detection Accuracy",
        3,
        10,
        5
    )

    st.markdown("---")

    st.subheader("🧠 AI Features")

    st.write("✅ Face Tracking")
    st.write("✅ Multiple Person Detection")
    st.write("✅ Real-time Alerts")
    st.write("✅ AI Surveillance")
    st.write("✅ Integrity Analytics")

# ============================================================
# METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">STATUS</div>
        <div class="metric-value">🟢</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">FACES</div>
        <div class="metric-value">{st.session_state.face_count}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">ALERTS</div>
        <div class="metric-value">{len(st.session_state.alerts)}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:

    score = max(
        100 - (len(st.session_state.alerts) * 8),
        0
    )

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">INTEGRITY</div>
        <div class="metric-value">{score}%</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ============================================================
# FACE DETECTOR
# ============================================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ============================================================
# AI VIDEO PROCESSOR
# ============================================================

class AIProctor(VideoTransformerBase):

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=sensitivity,
            minSize=(35,35)
        )

        st.session_state.face_count = len(faces)

        # Neon Overlay
        overlay = img.copy()

        cv2.rectangle(
            overlay,
            (0,0),
            (430,90),
            (15,23,42),
            -1
        )

        img = cv2.addWeighted(
            overlay,
            0.6,
            img,
            0.4,
            0
        )

        # Face Detection
        for (x,y,w,h) in faces:

            cv2.rectangle(
                img,
                (x,y),
                (x+w,y+h),
                (0,255,255),
                3
            )

            cv2.circle(
                img,
                (x+w//2,y+h//2),
                5,
                (255,0,255),
                -1
            )

            cv2.putText(
                img,
                "Verified Candidate",
                (x,y-12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2
            )

        # Alert System
        if monitoring:

            if len(faces) > allowed_faces:

                alert = (
                    f"⚠️ Suspicious Activity : "
                    f"{datetime.now().strftime('%H:%M:%S')}"
                )

                if alert not in st.session_state.alerts:
                    st.session_state.alerts.append(alert)

                cv2.putText(
                    img,
                    "SECURITY ALERT DETECTED",
                    (20,45),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (0,0,255),
                    3
                )

        # Overlay Text
        cv2.putText(
            img,
            f"Candidates : {len(faces)}",
            (20,35),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255,255,255),
            2
        )

        cv2.putText(
            img,
            datetime.now().strftime("%d %b %Y  %H:%M:%S"),
            (20,70),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255,255,255),
            2
        )

        return img

# ============================================================
# MAIN DASHBOARD
# ============================================================

left, right = st.columns([2.4,1])

# ============================================================
# LIVE CAMERA
# ============================================================

with left:

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("🎥 Live AI Surveillance")

    webrtc_streamer(
        key="ai-proctor",
        video_transformer_factory=AIProctor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# ALERTS PANEL
# ============================================================

with right:

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("🚨 Security Feed")

    if st.session_state.alerts:

        for item in reversed(st.session_state.alerts[-6:]):

            st.markdown(f"""
            <div class="alert">
                {item}
            </div>
            """, unsafe_allow_html=True)

    else:
        st.success("System Secure • No Violations")

    st.markdown("---")

    st.subheader("📈 AI Analytics")

    st.progress(score)

    exam_duration = int(
        time.time() - st.session_state.start_time
    )

    mins = exam_duration // 60
    secs = exam_duration % 60

    st.metric(
        "Session Time",
        f"{mins:02}:{secs:02}"
    )

    st.metric(
        "Integrity Score",
        f"{score}%"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.write("")

st.markdown("""
<center>
<p style='color:#64748b;'>
AI Exam Shield • Smart Proctoring Platform • Streamlit AI Dashboard
</p>
</center>
""", unsafe_allow_html=True)
