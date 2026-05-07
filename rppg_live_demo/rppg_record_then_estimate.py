import cv2
import numpy as np
import mediapipe as mp
import torch

from tiny_rppgnet import TinyRPPGNet
from rppg_core import bandpass_filter, bpm_from_welch_harmonic

# ===== Config =====
CLIP_SECONDS = 20          # choose 10, 15, or 20; longer = more stable
ROI_SIZE = (36, 36)
MODEL_CLIP_LEN = 128       # TinyRPPGNet expects 128 frames
MODEL_CLIP_STRIDE = 32     # stride between windows when averaging

mp_face_mesh = mp.solutions.face_mesh

def get_forehead_bbox_from_landmarks(landmarks, w, h, scale=1.05, frac=0.35):
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

def main():
    # ---- Load model ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyRPPGNet().to(device)
    state = torch.load("models/tiny_rppgnet_best.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[INFO] Loaded TinyRPPGNet on {device}")

    # ---- Open webcam ----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    target_frames = int(CLIP_SECONDS * fps)
    print(f"[INFO] Target frames: {target_frames} (fps ~ {fps:.1f})")

    roi_frames = []
    green_trace = []

    # ---- Record phase ----
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_bgr.shape[:2]

            results = face_mesh.process(frame_rgb)
            roi_ok = False

            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0]
                x1, y1, x2, y2 = get_forehead_bbox_from_landmarks(lms, w, h)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w - 1, x2); y2 = min(h - 1, y2)

                if x2 > x1 and y2 > y1:
                    roi = frame_rgb[y1:y2, x1:x2]
                    roi = cv2.resize(roi, ROI_SIZE)
                    roi_frames.append(roi)
                    g_val = roi[:, :, 1].mean()
                    green_trace.append(g_val)
                    roi_ok = True

                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2),
                                  (0, 255, 0), 1)

            # progress overlay
            n = len(roi_frames)
            progress = min(1.0, n / max(1, target_frames))
            cv2.rectangle(frame_bgr, (10, h-30), (10 + int((w-20)*progress), h-10),
                          (0, 255, 0), -1)
            cv2.putText(frame_bgr,
                        f"Recording... {progress*100:4.1f}% (hold still)",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

            cv2.imshow("rPPG Capture", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

            if n >= target_frames:
                break

    cap.release()
    cv2.destroyAllWindows()

    if len(roi_frames) < max(MODEL_CLIP_LEN, target_frames // 3):
        print("[ERROR] Not enough stable ROI frames captured.")
        return

    roi_frames = np.stack(roi_frames, axis=0)  # (T,H,W,3)
    green_trace = np.array(green_trace, dtype=np.float32)
    T = roi_frames.shape[0]
    print(f"[INFO] Captured {T} ROI frames.")

    # ---- Classical HR from entire clip ----
    trace_f = bandpass_filter(green_trace, fs=fps, low=0.7, high=4.0)
    bpm_classic = bpm_from_welch_harmonic(trace_f, fs=fps)

    # ---- TinyRPPGNet HR: average over sliding windows ----
    preds = []
    start = 0
    while start + MODEL_CLIP_LEN <= T:
        clip = roi_frames[start:start+MODEL_CLIP_LEN]  # (L,H,W,3)
        x = torch.from_numpy(clip).float() / 255.0
        x = x.permute(3,0,1,2).unsqueeze(0).to(device)  # (1,3,L,H,W)
        with torch.no_grad():
            pred = model(x).cpu().item()
        preds.append(pred)
        start += MODEL_CLIP_STRIDE

    bpm_tiny = float(np.mean(preds)) if preds else float("nan")

    print("\n===== Estimated Heart Rates =====")
    print(f"Classical rPPG (Welch, whole clip):   {bpm_classic:6.2f} bpm")
    print(f"TinyRPPGNet (avg over windows):       {bpm_tiny:6.2f} bpm")
    print("Compare with oximeter reading.")
    print("=================================\n")

if __name__ == "__main__":
    main()
