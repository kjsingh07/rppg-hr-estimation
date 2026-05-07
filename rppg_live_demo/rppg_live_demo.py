import cv2
import numpy as np
import mediapipe as mp
import torch

from tiny_rppgnet import TinyRPPGNet
from rppg_core import bandpass_filter, bpm_from_welch_harmonic

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

    # take top fraction as forehead
    h_box = y2 - y1
    y2_new = y1 + int(h_box * frac)
    return x1, y1, x2, max(y1 + 1, y2_new)

def main():
    # --------- Load model ---------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyRPPGNet().to(device)

    state = torch.load("models/tiny_rppgnet_best.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()

    print(f"[INFO] Loaded TinyRPPGNet best checkpoint on {device}")

    # --------- Init camera ---------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. "
              "Run this on a local machine with a working camera.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    roi_size = (36, 36)
    clip_len = 128
    roi_buffer = []
    trace_buffer = []

    mp_face_mesh = mp.solutions.face_mesh
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
            h, w = frame_rgb.shape[:2]

            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0]
                x1, y1, x2, y2 = get_forehead_bbox_from_landmarks(lms, w, h)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w - 1, x2); y2 = min(h - 1, y2)

                if x2 > x1 and y2 > y1:
                    roi = frame_rgb[y1:y2, x1:x2]
                    roi = cv2.resize(roi, roi_size)

                    # update buffers
                    roi_buffer.append(roi)
                    if len(roi_buffer) > clip_len:
                        roi_buffer.pop(0)

                    g_val = roi[:, :, 1].mean()
                    trace_buffer.append(g_val)
                    if len(trace_buffer) > clip_len * 2:
                        trace_buffer.pop(0)

                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2),
                                  (0, 255, 0), 1)

            # ----- Classical HR -----
            bpm_classic = float("nan")
            if len(trace_buffer) > 64:
                trace = np.array(trace_buffer, dtype=np.float32)
                trace_f = bandpass_filter(trace, fs=fps,
                                          low=0.7, high=4.0)
                bpm_classic = bpm_from_welch_harmonic(trace_f, fs=fps)

            # ----- TinyRPPGNet HR -----
            bpm_tiny = float("nan")
            if len(roi_buffer) >= clip_len:
                clip = np.stack(roi_buffer[-clip_len:], axis=0)  # (T,H,W,3)
                x = torch.from_numpy(clip).float() / 255.0
                x = x.permute(3, 0, 1, 2).unsqueeze(0).to(device)
                with torch.no_grad():
                    bpm_tiny = model(x).cpu().item()

            # ----- Overlay text -----
            line_y = 25
            if np.isfinite(bpm_classic):
                cv2.putText(frame_bgr,
                            f"Classic HR: {bpm_classic:5.1f} bpm",
                            (10, line_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2)
                line_y += 25

            if np.isfinite(bpm_tiny):
                cv2.putText(frame_bgr,
                            f"TinyRPPGNet HR: {bpm_tiny:5.1f} bpm",
                            (10, line_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 200, 255),
                            2)
                line_y += 25

            cv2.putText(frame_bgr,
                        "Press Q or ESC to quit",
                        (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1)

            cv2.imshow("Live rPPG Demo - TinyRPPGNet vs Classical",
                       frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
