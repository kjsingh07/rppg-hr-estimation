import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import torch
import time
from collections import deque

from tiny_rppgnet import TinyRPPGNet
from rppg_core import bandpass_filter, bpm_from_welch_harmonic

# Page configuration
st.set_page_config(
    page_title="Live Pulse Rate Monitor",
    page_icon="❤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive UI
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    div.stButton > button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        font-size: 18px;
        font-weight: bold;
        border: none;
        border-radius: 30px;
        padding: 15px 40px;
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 20px rgba(0, 0, 0, 0.4);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    h1 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.15);
        padding: 15px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'stop_requested' not in st.session_state:
    st.session_state.stop_requested = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None

def load_model():
    """Load the TinyRPPGNet model"""
    if not st.session_state.model_loaded:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TinyRPPGNet().to(device)
        try:
            state = torch.load("models/tiny_rppgnet_best.pth", map_location=device)
            model.load_state_dict(state)
            model.eval()
            st.session_state.model = model
            st.session_state.device = device
            st.session_state.model_loaded = True
            return True, device
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False, None
    return True, st.session_state.device

def get_forehead_bbox_from_landmarks(landmarks, w, h, scale=1.05, frac=0.35):
    """Extract forehead bounding box from face landmarks"""
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]

    x1 = max(int(min(xs) * w), 0)
    x2 = min(int(max(xs) * w), w - 1)
    y1 = max(int(min(ys) * h), 0)
    y2 = min(int(max(ys) * h), h - 1)

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale

    x1 = int(max(cx - bw / 2, 0))
    x2 = int(min(cx + bw / 2, w - 1))
    y1 = int(max(cy - bh / 2, 0))
    y2 = int(min(cy + bh / 2, h - 1))

    h_box = y2 - y1
    y2_new = y1 + int(h_box * frac)
    return x1, y1, x2, max(y1 + 1, y2_new)

def run_live_detection():
    """Run live video detection with real-time heart rate updates"""
    
    success, device = load_model()
    if not success:
        return
    
    model = st.session_state.model
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not access webcam. Please check your camera permissions.")
        st.session_state.running = False
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    
    roi_size = (36, 36)
    clip_len = 128
    roi_buffer = deque(maxlen=clip_len)
    trace_buffer = deque(maxlen=clip_len * 2)
    
    mp_face_mesh = mp.solutions.face_mesh
    
    # Create placeholders for UI elements
    video_placeholder = st.empty()
    col1, col2 = st.columns(2)
    metric_classic = col1.empty()
    metric_tiny = col2.empty()
    status_placeholder = st.empty()
    
    # Stop button outside the loop
    stop_col1, stop_col2, stop_col3 = st.columns([1, 2, 1])
    with stop_col2:
        stop_button = st.button("⏹ Stop Live Detection", type="secondary", use_container_width=True, key="stop_live")
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        
        frame_count = 0
        start_time = time.time()
        
        while st.session_state.running:
            # Check if stop button was clicked
            if stop_button:
                st.session_state.running = False
                break
            
            ok, frame_bgr = cap.read()
            if not ok:
                break
            
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            
            results = face_mesh.process(frame_rgb)
            face_detected = False
            
            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0]
                x1, y1, x2, y2 = get_forehead_bbox_from_landmarks(lms, w, h)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w - 1, x2); y2 = min(h - 1, y2)
                
                if x2 > x1 and y2 > y1:
                    face_detected = True
                    roi = frame_rgb[y1:y2, x1:x2]
                    roi_resized = cv2.resize(roi, roi_size)
                    
                    roi_buffer.append(roi_resized)
                    
                    g_val = roi_resized[:, :, 1].mean()
                    trace_buffer.append(g_val)
                    
                    # Draw rectangle and status
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame_bgr, "Face Detected", (10, 35),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            if not face_detected:
                cv2.putText(frame_bgr, "No Face Detected", (10, 35),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Calculate heart rates periodically
            if frame_count % 15 == 0 and len(trace_buffer) > 64:  # Update every 15 frames
                # Classical method
                bpm_classic = float("nan")
                try:
                    trace = np.array(list(trace_buffer), dtype=np.float32)
                    trace_f = bandpass_filter(trace, fs=fps, low=0.7, high=4.0)
                    bpm_classic = bpm_from_welch_harmonic(trace_f, fs=fps)
                except Exception as e:
                    pass
                
                # TinyRPPGNet method
                bpm_tiny = float("nan")
                if len(roi_buffer) >= clip_len:
                    try:
                        clip = np.stack(list(roi_buffer)[-clip_len:], axis=0)
                        x = torch.from_numpy(clip).float() / 255.0
                        x = x.permute(3, 0, 1, 2).unsqueeze(0).to(device)
                        with torch.no_grad():
                            bpm_tiny = model(x).cpu().item()
                    except Exception as e:
                        pass
                
                # Update metrics
                if np.isfinite(bpm_classic):
                    metric_classic.metric("🔵 Classical Method", f"{bpm_classic:.1f} BPM")
                    cv2.putText(frame_bgr, f"Classic: {bpm_classic:.1f} BPM", (10, h - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    metric_classic.metric("🔵 Classical Method", "Collecting data...")
                
                if np.isfinite(bpm_tiny):
                    metric_tiny.metric("🟣 TinyRPPGNet", f"{bpm_tiny:.1f} BPM")
                    cv2.putText(frame_bgr, f"TinyRPPG: {bpm_tiny:.1f} BPM", (10, h - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                else:
                    metric_tiny.metric("🟣 TinyRPPGNet", "Collecting data...")
            
            # Add FPS and buffer info
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame_bgr, f"FPS: {fps_actual:.1f} | Frames: {len(roi_buffer)}/{clip_len}", 
                       (w - 350, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            video_placeholder.image(frame_bgr, channels="BGR", use_container_width=True)
            
            frame_count += 1
            
            # Small delay to control frame rate
            time.sleep(0.033)  # ~30 FPS
    
    cap.release()
    st.session_state.running = False
    status_placeholder.success("✅ Live detection stopped")

# Main UI
st.title("❤ Live Pulse Rate Monitor")
st.markdown("### Real-time Heart Rate Detection using rPPG Technology")

# Sidebar
with st.sidebar:
    st.header("⚙ Settings")
    st.markdown("---")
    
    st.markdown("### 📊 How it works")
    st.info("""
    1. Click *Start Live Detection*
    2. Position your face in the camera
    3. Stay still for best results
    4. Heart rate updates in real-time
    5. Click *Stop* to end detection
    """)
    
    st.markdown("---")
    st.markdown("### 🔬 Technology")
    st.success("""
    - *Classical Method*: Signal processing with FFT & Welch PSD
    - *TinyRPPGNet*: Deep learning 3D CNN model
    - *rPPG*: Remote photoplethysmography
    - *Real-time*: Live webcam processing
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Tips for Best Results")
    st.warning("""
    - *Good lighting* is essential
    - Keep your *face steady*
    - Wait *5-10 seconds* for accurate reading
    - *Avoid sudden movements*
    - Ensure your *forehead is visible*
    """)
    
    st.markdown("---")
    st.markdown("### 📋 Normal Heart Rate Ranges")
    st.info("""
    - *Resting*: 60-100 BPM
    - *Athlete*: 40-60 BPM
    - *After exercise*: 100-150 BPM
    """)

# Main content area
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    
    if not st.session_state.running:
        if st.button("🎥 Start Live Detection", use_container_width=True, type="primary"):
            st.session_state.running = True
            st.rerun()
    else:
        st.info("🔴 Live detection is running...")

# Run live detection if active
if st.session_state.running:
    run_live_detection()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <p>🏥 <b>Disclaimer:</b> This is for educational purposes only. Not for medical diagnosis.</p>
    <p>💻 Powered by TinyRPPGNet, MediaPipe & PyTorch</p>
    <p>🎓 Remote Photoplethysmography (rPPG) Demo</p>
</div>
""", unsafe_allow_html=True)