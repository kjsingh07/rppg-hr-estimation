# Remote Photoplethysmography (rPPG) for Contactless Heart Rate Estimation with Classical and Deep Learning Pipelines

## Overview
This project is a contactless heart rate estimation system based on remote photoplethysmography (rPPG). It extracts physiological pulse information from facial video and estimates heart rate without physical sensors.

The repository includes classical signal-processing methods, a lightweight deep learning model, and a Streamlit-based demo interface for interactive use.

## Features
- Facial region-of-interest extraction using MediaPipe Face Mesh
- Classical rPPG pipelines using:
  - Green channel method
  - CHROM
  - POS
- Signal filtering and frequency-domain BPM estimation
- Custom lightweight PyTorch model for heart-rate regression
- Live webcam-based inference
- Record-then-estimate workflow
- Streamlit demo for a simple user interface

## Workflow
1. Capture facial video from webcam or input file
2. Detect facial landmarks and extract a stable forehead ROI
3. Generate rPPG traces from the ROI
4. Apply temporal filtering and BPM estimation
5. Compare classical estimation with the learned model

## Project Components
- `rppg_core.py`  
  Core signal-processing and ROI extraction utilities

- `tiny_rppgnet.py`  
  Lightweight deep learning model for BPM regression

- `rppg_live_demo.py`  
  Live webcam demo for real-time heart-rate estimation

- `rppg_record_then_estimate.py`  
  Workflow for recording a clip and then estimating BPM

- `rppg_unified_demo.py`  
  Combined interface for running multiple modes

- `app.py`  
  Streamlit app for interactive demo use

- `models/tiny_rppgnet_best.pth`  
  Saved model checkpoint

- `requirements.txt`  
  Python dependencies

## Dataset
The project was developed and evaluated using the **UBFC-rPPG dataset**, which provides synchronized facial video and ground-truth pulse signals.

## Methodology
### Classical rPPG
The project implements multiple classical approaches for pulse extraction:
- Green channel averaging
- CHROM-based signal construction
- POS-based signal construction

### Signal Processing
- Bandpass filtering
- Detrending and smoothing
- FFT and Welch-based BPM estimation

### Deep Learning
A custom lightweight 3D CNN was trained to predict heart rate from short video clips. The model is designed to work on spatiotemporal facial data and complement the classical pipeline.

## Tech Stack
- Python
- OpenCV
- MediaPipe
- NumPy
- SciPy
- PyTorch
- Matplotlib
- Streamlit

## Results
The system supports both classical and learned heart-rate estimation and is designed for short-duration webcam recordings. It was also calibrated for varied skin tones, including Indian skin tones, to improve practical robustness.
