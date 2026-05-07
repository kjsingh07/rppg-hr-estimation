# rppg_unified_demo.py
# One-file live rPPG demo with a chroma bar overlay:
# - Records N seconds from webcam
# - Shows a small bar (top-left) that visualizes amplified per-frame color changes
#   from the forehead ROI (what the camera sees but eyes don't)
# - Classical HR using POS/CHROM/GREEN (best-by-SNR)
# - TinyRPPGNet HR via median of sliding windows
# - NO calibration offset

import os
import cv2
import time
import math
import argparse
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal

# =========================
# Model definition
# =========================
class TinyRPPGNet(nn.Module):
    """
    Lightweight 3D CNN for rPPG-based HR regression.
    Input : (B, 3, T, H, W)
    Output: (B,) BPM
    """
    def __init__(self, in_ch=3):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, 16, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2))
        self.bn1   = nn.BatchNorm3d(16)

        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))
        self.bn2   = nn.BatchNorm3d(32)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))
        self.bn3   = nn.BatchNorm3d(64)

        self.conv4 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.bn4   = nn.BatchNorm3d(64)

        self.pool  = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc1   = nn.Linear(64, 32)
        self.drop  = nn.Dropout(0.3)
        self.fc2   = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x).view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x).squeeze(1)
        return x

# =========================
# Signal processing helpers
# =========================
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.999999)
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(sig, fs, low=0.7, high=4.0, order=3):
    sig = np.asarray(sig, dtype=np.float32)
    if len(sig) < 20:
        return sig
    sig = sig - np.nanmean(sig)
    sig = signal.detrend(sig)
    b, a = butter_bandpass(low, high, fs, order)
    return signal.filtfilt(b, a, sig)

def bpm_from_welch_harmonic(sig, fs, f_lo=0.7, f_hi=4.0):
    """Estimate BPM via Welch with simple harmonic sanity (avoid 1/2x & 2x)."""
    sig = np.asarray(sig, dtype=np.float32)
    if len(sig) < 32:
        return float("nan")
    sig = signal.detrend(sig - np.mean(sig))
    freqs, psd = signal.welch(sig, fs=fs, nperseg=min(len(sig), 256))
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        return float("nan")
    freqs = freqs[mask]; psd = psd[mask]
    base_idx = np.argmax(psd); f0 = freqs[base_idx]

    def local_power(f_target, tol=0.08):
        m = (freqs >= f_target*(1-tol)) & (freqs <= f_target*(1+tol))
        return psd[m].max() if np.any(m) else 0.0

    p_base  = local_power(f0)
    p_half  = local_power(f0/2)
    p_double= local_power(f0*2)

    best = f0
    if p_half > p_base * 1.2:   best = f0 / 2.0
    if p_double > max(p_base, p_half) * 1.2: best = f0 * 2.0
    return float(best * 60.0)

# POS / CHROM / GREEN traces
def green_trace(frames_rgb):
    return frames_rgb[:,:,:,1].astype(np.float32).mean(axis=(1,2))

def pos_trace(frames_rgb):
    X = frames_rgb.astype(np.float32) / 255.0
    R = X[:,:,:,0].mean(axis=(1,2))
    G = X[:,:,:,1].mean(axis=(1,2))
    B = X[:,:,:,2].mean(axis=(1,2))
    C = np.vstack([R, G, B])  # (3,T)
    Cn = (C - C.mean(axis=1, keepdims=True)) / (C.std(axis=1, keepdims=True) + 1e-8)
    S1 = Cn[1] - Cn[2]                    # G - B
    S2 = Cn[1] + Cn[2] - 2*Cn[0]          # G + B - 2R
    alpha = np.std(S1) / (np.std(S2) + 1e-8)
    s = S1 - alpha * S2
    return s

def chrom_trace(frames_rgb):
    X = frames_rgb.astype(np.float32) / 255.0
    R = X[:,:,:,0].mean(axis=(1,2))
    G = X[:,:,:,1].mean(axis=(1,2))
    B = X[:,:,:,2].mean(axis=(1,2))
    C = np.vstack([R, G, B])  # (3,T)
    C = C - C.mean(axis=1, keepdims=True)
    C = C / (C.std(axis=1, keepdims=True) + 1e-8)
    X1 = 3*C[0] - 2*C[1]
    X2 = 1.5*C[0] + C[1] - 1.5*C[2]
    s = X1 / (X2 + 1e-8)
    return s

def snr_of(sig, fs, f_lo=0.7, f_hi=4.0):
    freqs, psd = signal.welch(sig, fs=fs, nperseg=min(len(sig),256))
    def pwr(fa, fb):
        m = (freqs >= fa) & (freqs <= fb)
        return psd[m].sum() if np.any(m) else 1e-12
    signal_band = pwr(f_lo, f_hi)
    noise_band  = pwr(0.2, 0.6) + pwr(4.5, 5.0)
    return float(signal_band / (noise_band + 1e-12))

# =========================
# Face ROI (forehead) via MediaPipe
# =========================
mp_face_mesh = mp.solutions.face_mesh

def get_forehead_bbox_from_landmarks(landmarks, w, h, scale=1.05, frac=0.35):
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    x1 = max(int(min(xs) * w), 0); x2 = min(int(max(xs) * w), w - 1)
    y1 = max(int(min(ys) * h), 0); y2 = min(int(max(ys) * h), h - 1)

    cx = 0.5 * (x1 + x2); cy = 0.5 * (y1 + y2)
    bw = (x2 - x1) * scale; bh = (y2 - y1) * scale
    x1 = int(max(cx - bw / 2, 0)); x2 = int(min(cx + bw / 2, w - 1))
    y1 = int(max(cy - bh / 2, 0)); y2 = int(min(cy + bh / 2, h - 1))

    # take top fraction as forehead
    h_box = y2 - y1
    y2_new = y1 + int(h_box * frac)
    return x1, y1, x2, max(y1 + 1, y2_new)

# =========================
# Main: record then estimate (+ chroma bar)
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=20, help="Clip length to record")
    ap.add_argument("--clip_len", type=int, default=128, help="Frames per model window")
    ap.add_argument("--stride", type=int, default=32, help="Stride between model windows")
    ap.add_argument("--roi_w", type=int, default=36, help="ROI width")
    ap.add_argument("--roi_h", type=int, default=36, help="ROI height")
    ap.add_argument("--model", type=str, default="models/tiny_rppgnet_best.pth", help="Model checkpoint path")
    # chroma bar params
    ap.add_argument("--bar_width", type=int, default=220, help="Chroma bar width in px")
    ap.add_argument("--bar_height", type=int, default=24, help="Chroma bar height in px")
    ap.add_argument("--bar_amp", type=float, default=6.0, help="Amplification of per-frame color change")
    ap.add_argument("--bar_alpha", type=float, default=0.95, help="EMA smoothing for baseline color (0..1)")
    args = ap.parse_args()

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyRPPGNet().to(device)
    if not os.path.exists(args.model):
        print(f"[ERROR] Model checkpoint not found: {args.model}")
        return
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[INFO] Loaded TinyRPPGNet on {device}")

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    target_frames = int(args.seconds * fps)
    print(f"[INFO] Recording ~{args.seconds}s ({target_frames} frames @ ~{fps:.1f} fps). Hold still & keep good light.")

    roi_frames = []
    Hroi, Wroi = args.roi_h, args.roi_w

    # --- chroma bar state ---
    bar_h, bar_w = args.bar_height, args.bar_width
    chroma_bar = np.zeros((bar_h, bar_w, 3), dtype=np.uint8)
    ema_rgb = None  # exponential moving average of mean ROI color

    # Record phase
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
            mean_rgb = None

            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0]
                x1, y1, x2, y2 = get_forehead_bbox_from_landmarks(lms, w, h)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w - 1, x2); y2 = min(h - 1, y2)

                if x2 > x1 and y2 > y1:
                    roi = frame_rgb[y1:y2, x1:x2]
                    roi = cv2.resize(roi, (Wroi, Hroi))
                    roi_frames.append(roi)
                    roi_ok = True
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0,255,0), 1)

                    # --- chroma bar update (amplified subtle color change) ---
                    # mean RGB of current ROI
                    mean_rgb = roi.reshape(-1,3).mean(axis=0).astype(np.float32)  # [R,G,B] in 0..255
                    if ema_rgb is None:
                        ema_rgb = mean_rgb.copy()
                    else:
                        ema_rgb = args.bar_alpha * ema_rgb + (1.0 - args.bar_alpha) * mean_rgb

                    # amplify deviation from baseline (EMA)
                    amplified = ema_rgb + args.bar_amp * (mean_rgb - ema_rgb)
                    amplified = np.clip(amplified, 0, 255).astype(np.uint8)  # RGB

                    # shift bar left by 1 and draw new column (convert to BGR for display)
                    chroma_bar[:, 0:-1] = chroma_bar[:, 1:]
                    chroma_bar[:, -1] = amplified[::-1]  # RGB->BGR

            # progress bar
            n = len(roi_frames)
            progress = min(1.0, n / max(1, target_frames))
            cv2.rectangle(frame_bgr, (10, h-30), (10 + int((w-20)*progress), h-10), (0,255,0), -1)
            cv2.putText(frame_bgr, f"Recording... {progress*100:4.1f}%", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # overlay chroma bar (top-left)
            y0, x0 = 10, 10
            y1, x1 = y0 + bar_h, x0 + bar_w
            if y1 < h and x1 < w:
                # paste bar
                frame_bgr[y0:y1, x0:x1] = chroma_bar
                # border + label
                cv2.rectangle(frame_bgr, (x0-1, y0-1), (x1+1, y1+1), (255,255,255), 1)
                cv2.putText(frame_bgr, "chroma (amplified)", (x0, y0 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

            cv2.imshow("rPPG Capture", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if n >= target_frames:
                break

    cap.release()
    cv2.destroyAllWindows()

    if len(roi_frames) < max(args.clip_len, target_frames // 3):
        print("[ERROR] Not enough stable ROI frames captured.")
        return

    roi_frames = np.stack(roi_frames, axis=0)  # (T,H,W,3)
    T = roi_frames.shape[0]
    print(f"[INFO] Captured {T} ROI frames.")

    # Build & filter candidate traces for the whole clip
    tr_green = bandpass_filter(green_trace(roi_frames), fs=fps)
    tr_pos   = bandpass_filter(pos_trace(roi_frames),   fs=fps)
    tr_chrom = bandpass_filter(chrom_trace(roi_frames), fs=fps)

    traces = {"GREEN": tr_green, "POS": tr_pos, "CHROM": tr_chrom}
    snrs = {k: snr_of(v, fs=fps) for k, v in traces.items()}
    best_name = max(snrs, key=snrs.get)
    best_trace = traces[best_name]

    # Classical HR from best trace
    bpm_classic = bpm_from_welch_harmonic(best_trace, fs=fps)

    # TinyRPPGNet: median over sliding windows
    preds = []
    start = 0
    while start + args.clip_len <= T:
        clip = roi_frames[start:start+args.clip_len]  # (L,H,W,3)
        x = torch.from_numpy(clip).float() / 255.0
        x = x.permute(3,0,1,2).unsqueeze(0).to(device)  # (1,3,L,H,W)
        with torch.no_grad():
            pred = model(x).cpu().item()
        preds.append(pred)
        start += args.stride

    bpm_tiny = float(np.median(preds)) if preds else float("nan")

    print("\n===== Estimated Heart Rates =====")
    print(f"Trace chosen: {best_name} (SNR: {snrs[best_name]:.2f})")
    print(f"Classical rPPG (Welch, whole clip):   {bpm_classic:6.2f} bpm")
    print(f"TinyRPPGNet (median over windows):    {bpm_tiny:6.2f} bpm")
    print("Compare with oximeter reading for validation.")
    print("=================================\n")

if __name__ == "__main__":
    main()
