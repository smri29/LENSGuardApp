🛡️ LENS-Guard: Hybrid Network Intrusion Detection System

LENS-Guard (Lightweight Ensemble Network Security Guard) is a state-of-the-art Hybrid Network Intrusion Detection System (NIDS). It leverages a dual-engine architecture combining Supervised Learning (1D-CNN) and Unsupervised Learning (Variational Autoencoder) to detect known cyberattacks and zero-day anomalies with 99% accuracy.

This repository contains the deployment code for the LENS-Guard Web Dashboard, featuring a premium "Command Center" UI for real-time security monitoring.

🧠 System Architecture

LENS-Guard employs a sophisticated "Ensemble" logic where an alert is triggered if either stage detects a threat:

Stage A (Signature Engine): A lightweight 1D-CNN trained to identify known attack patterns (e.g., DoS, PortScan, Botnet).

Stage B (Anomaly Engine): A Variational Autoencoder (VAE) trained exclusively on benign traffic. It flags any traffic with a high reconstruction error as a potential zero-day threat.

Hybrid Consensus: Threat = (CNN == Attack) OR (VAE > Threshold)

📂 Project Structure

Ensure your project folder looks exactly like this:

LENS_Guard_App/
├── app.py                          # Main Streamlit application (Command Center UI)
├── requirements.txt                # Python dependencies
├── student_cnn.keras               # Pre-trained 1D-CNN Model (Stage A)
├── vae_weights.weights.h5          # Pre-trained VAE Weights (Stage B)
├── scaler.pkl                      # Fitted MinMaxScaler for preprocessing
├── config.json                     # System configuration (Thresholds, feature counts)
├── lens_guard_test_sample_500.csv  # Sample dataset (500 flows) for testing
├── lens_guard_test_sample_1000.csv # Sample dataset (1000 flows) for testing
└── README.md                       # Documentation


🚀 Quick Start Guide

1. Installation

Clone the repository and install the required dependencies:

git clone [https://github.com/smri29/LENSGuardApp.git](https://github.com/smri29/LENSGuardApp.git)
cd LENSGuardApp
pip install -r requirements.txt


2. Launch the Dashboard

Run the application locally:

streamlit run app.py


The application will open automatically in your browser at http://localhost:8501.

💻 Usage & Testing

🟢 1. Landing Page (No File Uploaded)

When you first launch the app, you will see the Welcome Screen.

Download Test Data: You can download the included sample files (500 Flows or 1000 Flows) directly from the dashboard to test the system immediately.

Template: A blank template is also provided if you want to map your own data.

🔴 2. Real-Time Analysis (File Uploaded)

Upload a CSV file to the sidebar. The system will perform:

Automated Preprocessing: Converts categorical data and scales numerical features.

Dual-Core Scanning: Runs both CNN and VAE models in parallel.

Visualization:

Traffic Overview: Donut charts showing the ratio of Safe vs. Malicious traffic.

Anomaly Timeline: Interactive scatter plot pinpointing exactly when attacks occurred.

Deep Dive: Histograms of reconstruction errors.

⚙️ 3. Sensitivity Tuning

Use the slider in the Control Panel to adjust the Anomaly Threshold in real-time. Lowering the threshold makes the system stricter (higher security), while raising it reduces false positives.

📊 Data Requirements

To use your own data, your CSV file must match the BCCC-CIC-IDS2017 feature set:

Columns: Exactly 118 numerical feature columns.

Protocol Columns: proto_TCP and proto_UDP must be integers (0 or 1), not booleans.

☁️ Deployment

LENS-Guard is optimized for Streamlit Cloud:

Push this repository to GitHub.

Log in to share.streamlit.io.

Select your repo and deploy.

Note: Ensure student_cnn.keras and weights are uploaded via Git LFS if they exceed 100MB.

👨‍💻 Developer

Developed by Shah Mohammad Rizvi

LENS-Guard v1.0 Production Release

Specialized in AI-driven Cybersecurity Solutions.

For issues or contributions, please open a pull request.
