import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Reshape, Conv1DTranspose, BatchNormalization, Lambda, Cropping1D
from tensorflow.keras import backend as K
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
import io

# --- 1. Page Configuration (Must be first) ---
st.set_page_config(
    page_title="LENS-Guard AI Defense",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Command Center Look ---
st.markdown("""
<style>
    /* 1. Main Background - Deep Radial Gradient */
    .stApp {
        background: radial-gradient(ellipse at center, #1e293b 0%, #020617 100%);
        color: #e2e8f0;
    }

    /* 2. Typography */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #ffffff !important;
    }
    h1 { font-weight: 800; letter-spacing: -1px; background: -webkit-linear-gradient(#22d3ee, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    p, div, label { color: #cbd5e1; }

    /* 3. Glassmorphism Cards (Metrics) */
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
        padding: 15px;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: #22d3ee;
        box-shadow: 0 10px 30px rgba(6, 182, 212, 0.15);
    }
    
    /* 4. Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }

    /* 5. Custom Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: rgba(15, 23, 42, 0.9);
        backdrop-filter: blur(10px);
        color: #94a3b8;
        text-align: center;
        padding: 12px;
        border-top: 1px solid #1e293b;
        font-size: 13px;
        letter-spacing: 0.5px;
        z-index: 999;
    }
    
    /* 6. Buttons */
    div.stDownloadButton > button {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #3b82f6;
        color: #60a5fa;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    div.stDownloadButton > button:hover {
        background: #3b82f6;
        color: white;
        border-color: #3b82f6;
    }
</style>
<div class="footer">
    🛡️ <b>LENS-Guard</b> | Engineered by <b>Shah Mohammad Rizvi</b> | v1.0 Production
</div>
""", unsafe_allow_html=True)

# --- 2. Model Architecture Definitions ---

def sampling(args):
    """Reparameterization trick for VAE."""
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        return reconstruction

def build_vae(n_features=118, latent_dim=16):
    """Rebuilds the VAE architecture."""
    inputs_vae = Input(shape=(n_features, 1))
    x = Conv1D(32, 3, activation='relu', padding='same', strides=2)(inputs_vae)
    x = BatchNormalization()(x)
    x = Conv1D(64, 3, activation='relu', padding='same', strides=2)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs_vae, [z_mean, z_log_var, z], name='encoder')

    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(30 * 64, activation='relu')(latent_inputs)
    x = Reshape((30, 64))(x)
    x = Conv1DTranspose(32, 3, activation='relu', padding='same', strides=2)(x)
    x = BatchNormalization()(x)
    x = Conv1DTranspose(1, 3, activation='sigmoid', padding='same', strides=2)(x)
    x = Cropping1D(cropping=(1, 1))(x) 
    outputs_vae = x
    decoder = Model(latent_inputs, outputs_vae, name='decoder')
    
    vae = VAE(encoder, decoder)
    return vae

# --- 3. Resource Loading (Cached) ---
@st.cache_resource
def load_system_resources():
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    scaler = joblib.load('scaler.pkl')
    cnn_model = load_model('student_cnn.keras', compile=False)
    
    vae_model = build_vae(n_features=config['n_features'])
    vae_model.predict(np.zeros((1, config['n_features'], 1)), verbose=0)
    vae_model.load_weights('vae_weights.weights.h5')
    
    return cnn_model, vae_model, scaler, config

# --- 4. Main Application ---

# Header with Logo
c1, c2 = st.columns([1, 8])
with c1:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
with c2:
    st.title("LENS-Guard Defense System")
    st.markdown("### ⚡ Hybrid 1D-CNN + VAE Network Intrusion Detection")

# Load Models
try:
    with st.spinner("🔄 Initializing Neural Defense Cores..."):
        cnn, vae, scaler, config = load_system_resources()
        TRAINED_THRESHOLD = config['threshold']
except Exception as e:
    st.error(f"Critical System Failure: {e}")
    st.stop()

# Sidebar
st.sidebar.header("🎛️ Control Center")
uploaded_file = st.sidebar.file_uploader("Upload Network Logs (CSV)", type=['csv'])

st.sidebar.markdown("---")
st.sidebar.subheader("🎚️ Sensitivity Tuning")
use_custom_threshold = st.sidebar.checkbox("Override VAE Threshold", value=False)
if use_custom_threshold:
    current_threshold = st.sidebar.slider("Anomaly Threshold (MSE)", 0.0, 20.0, TRAINED_THRESHOLD)
else:
    current_threshold = TRAINED_THRESHOLD
    st.sidebar.info(f"System Threshold: {current_threshold:.4f}")

st.sidebar.markdown("---")
st.sidebar.caption("🟢 System Status: **Online**")
st.sidebar.caption("🧠 Models: **Loaded**")

# --- Logic: No File Uploaded (Instructions) ---
if uploaded_file is None:
    st.markdown("---")
    
    # Hero Section
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.5); padding: 20px; border-radius: 10px; border-left: 5px solid #3b82f6;">
        <h3 style="margin:0;">👋 Welcome to LENS-Guard</h3>
        <p style="margin-top:10px;">
            This system utilizes a dual-engine AI architecture to secure network infrastructure. 
            <b>Stage A (CNN)</b> detects known attack signatures, while <b>Stage B (VAE)</b> identifies zero-day anomalies.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📥 Test Data Assets")
    
    c_dl1, c_dl2 = st.columns(2)
    
    # Button for 500 samples
    with c_dl1:
        st.info("**Sample Set A (Light)**")
        try:
            with open("lens_guard_test_sample_500.csv", "rb") as f:
                st.download_button(
                    label="📄 Download 500 Flows",
                    data=f,
                    file_name="lens_guard_sample_500.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        except FileNotFoundError:
            st.error("Missing: lens_guard_test_sample_500.csv")

    # Button for 1000 samples
    with c_dl2:
        st.info("**Sample Set B (Heavy)**")
        try:
            with open("lens_guard_test_sample_1000.csv", "rb") as f:
                st.download_button(
                    label="📄 Download 1000 Flows",
                    data=f,
                    file_name="lens_guard_sample_1000.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        except FileNotFoundError:
            st.error("Missing: lens_guard_test_sample_1000.csv")

    st.markdown("---")
    st.caption("Please upload a CSV file containing exactly 118 feature columns (BCCC-CIC-IDS2017 Standard).")

# --- Logic: File Uploaded (Analysis) ---
else:
    try:
        # Load Data
        df = pd.read_csv(uploaded_file)
        
        # Validation
        bool_cols = df.select_dtypes(include=['bool']).columns
        if len(bool_cols) > 0:
            df[bool_cols] = df[bool_cols].astype(int)

        df_numeric = df.select_dtypes(include=[np.number])
        
        if 'label' in df_numeric.columns:
            df_numeric = df_numeric.drop(columns=['label'])
            
        if df_numeric.shape[1] != 118:
            st.error(f"❌ Dimension Mismatch. Expected 118 features, found {df_numeric.shape[1]}.")
            st.stop()

        # Processing UI
        with st.status("🚀 Processing Network Traffic...", expanded=True) as status:
            st.write("Encoding and Scaling Data...")
            X_scaled = scaler.transform(df_numeric)
            X_input = X_scaled.reshape(-1, 118, 1)
            
            st.write("Running Stage A: Convolutional Neural Network (Signatures)...")
            cnn_probs = cnn.predict(X_input, verbose=0).flatten()
            cnn_preds = (cnn_probs >= 0.5).astype(int)
            
            st.write("Running Stage B: Variational Autoencoder (Anomalies)...")
            reconstructions = vae.predict(X_input, verbose=0)
            mse_errors = np.sum(np.square(X_input - reconstructions), axis=(1, 2))
            vae_preds = (mse_errors > current_threshold).astype(int)
            
            st.write("Aggregating Hybrid Logic...")
            final_preds = np.logical_or(cnn_preds == 1, vae_preds == 1).astype(int)
            status.update(label="✅ Analysis Complete", state="complete", expanded=False)
        
        # Prepare Results
        results_df = df.copy()
        results_df['LENS_Status'] = ['Threat' if x == 1 else 'Safe' for x in final_preds]
        results_df['Confidence'] = cnn_probs
        results_df['Anomaly_Score'] = mse_errors
        results_df['Detection_Source'] = [
            'Signature (CNN)' if (c==1 and v==0) else 
            'Anomaly (VAE)' if (v==1 and c==0) else 
            'Critical (Both)' if (v==1 and c==1) else 
            'Safe' 
            for c, v in zip(cnn_preds, vae_preds)
        ]

        # --- Dashboard UI ---
        st.markdown("### 🛡️ Security Audit Report")
        
        # KPI Row
        total_threats = np.sum(final_preds)
        threat_rate = (total_threats / len(df)) * 100
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Flows Scanned", len(df))
        k2.metric("Threats Detected", int(total_threats), delta_color="inverse")
        k3.metric("Safe Traffic", len(df) - int(total_threats))
        k4.metric("Threat Percentage", f"{threat_rate:.2f}%", delta_color="inverse")
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Traffic Overview", "📉 Anomaly Deep Dive", "📋 Raw Logs"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                # Donut Chart with Custom Colors
                fig_donut = px.pie(
                    results_df, names='Detection_Source', 
                    title='Threat Classification',
                    color='Detection_Source',
                    color_discrete_map={
                        'Safe': '#10b981',
                        'Signature (CNN)': '#f43f5e',
                        'Anomaly (VAE)': '#a855f7',
                        'Critical (Both)': '#f97316'
                    },
                    hole=0.5
                )
                fig_donut.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
                st.plotly_chart(fig_donut, use_container_width=True)
                
            with col2:
                # VAE Histogram
                fig_hist = px.histogram(
                    x=mse_errors, nbins=60, 
                    title='Reconstruction Error Distribution (MSE)',
                    labels={'x': 'MSE Loss'},
                    color_discrete_sequence=['#3b82f6']
                )
                fig_hist.add_vline(x=current_threshold, line_dash="dash", line_color="#ef4444", annotation_text="Threshold")
                fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
                st.plotly_chart(fig_hist, use_container_width=True)

        with tab2:
            st.subheader("🔎 Anomaly Timeline")
            
            # Interactive Scatter Plot
            fig_scatter = go.Figure()
            
            # Safe points
            safe_mask = mse_errors <= current_threshold
            fig_scatter.add_trace(go.Scatter(
                x=np.where(safe_mask)[0], 
                y=mse_errors[safe_mask],
                mode='markers', name='Safe',
                marker=dict(color='#10b981', size=4, opacity=0.5)
            ))
            
            # Anomaly points
            ano_mask = mse_errors > current_threshold
            fig_scatter.add_trace(go.Scatter(
                x=np.where(ano_mask)[0], 
                y=mse_errors[ano_mask],
                mode='markers', name='Anomaly (Threat)',
                marker=dict(color='#f43f5e', size=6, opacity=0.9)
            ))
            
            # Threshold Line
            fig_scatter.add_hline(y=current_threshold, line_dash="dash", line_color="#ef4444", annotation_text="Threshold")
            
            fig_scatter.update_layout(
                xaxis_title="Flow Index (Time)",
                yaxis_title="Reconstruction Error (MSE)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=500
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with tab3:
            st.subheader("📄 Detailed Security Logs")
            
            # Filter
            filter_opt = st.radio("View Filter:", ["All", "Threats Only", "Safe Only"], horizontal=True)
            if filter_opt == "Threats Only":
                view_df = results_df[results_df['LENS_Status'] == 'Threat']
            elif filter_opt == "Safe Only":
                view_df = results_df[results_df['LENS_Status'] == 'Safe']
            else:
                view_df = results_df
                
            st.dataframe(view_df.head(2000), height=400) 
            
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Security Audit (CSV)",
                data=csv_data,
                file_name="LENS_Guard_Full_Report.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Analysis Error: {e}")