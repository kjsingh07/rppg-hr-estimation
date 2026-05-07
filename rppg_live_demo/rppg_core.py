import numpy as np
from scipy import signal

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(sig, fs, low=0.7, high=4.0, order=3):
    """
    Band-pass filter for rPPG signals (0.7–4 Hz).
    """
    sig = np.asarray(sig, dtype=np.float32)
    sig = sig - np.nanmean(sig)
    sig = np.nan_to_num(sig)
    if len(sig) < 10:
        return sig
    b, a = butter_bandpass(low, high, fs, order)
    return signal.filtfilt(b, a, sig)

def bpm_from_welch_harmonic(sig, fs, f_lo=0.7, f_hi=4.0):
    """
    Estimate BPM from signal using Welch PSD with simple harmonic sanity.
    Tries to avoid 1/2x or 2x errors.
    """
    sig = np.asarray(sig, dtype=np.float32)
    if len(sig) < 32:
        return float("nan")

    sig = signal.detrend(sig - np.mean(sig))
    freqs, psd = signal.welch(sig, fs=fs, nperseg=min(len(sig), 256))
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        return float("nan")

    freqs = freqs[mask]
    psd = psd[mask]
    base_idx = np.argmax(psd)
    f0 = freqs[base_idx]

    def local_power(f_target, tol=0.08):
        m = (freqs >= f_target*(1-tol)) & (freqs <= f_target*(1+tol))
        return psd[m].max() if np.any(m) else 0.0

    p_base = local_power(f0)
    p_half = local_power(f0/2)
    p_double = local_power(f0*2)

    best = f0
    if p_half > p_base * 1.2:
        best = f0 / 2.0
    if p_double > max(p_base, p_half) * 1.2:
        best = f0 * 2.0

    return float(best * 60.0)
