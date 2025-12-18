import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import joblib
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# =============================
# Page config
# =============================
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎧",
    layout="wide"
)

# =============================
# Light UI Theme (CSS)
# =============================
st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }
    h1, h2, h3, h4 {
        color: #0f172a;
    }
    .stMetric {
        background-color: #e0f2fe;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #bae6fd;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================
# Load models & evaluation (ONCE)
# =============================
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "emotion_mlp_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "feature_scaler.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

with open(os.path.join(BASE_DIR, "evaluation_results.pkl"), "rb") as f:
    eval_results = pickle.load(f)

eval_accuracy = eval_results["accuracy"]
conf_matrix = eval_results["confusion_matrix"]
labels = eval_results["labels"]

# =============================
# Audio processing
# =============================
def process_audio(file, n_mfcc=13):
    y, sr = librosa.load(file, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    return y, sr, duration, mfcc_mean, mel_db

# =============================
# Sidebar
# =============================
st.sidebar.title("ℹ️ About the App")
st.sidebar.markdown(
    """
This app predicts **human emotion from speech audio** using:

- ✅ MFCC audio features  
- ✅ Trained Neural Network (MLP)

### How to Use:
1. Upload a `.wav` file  
2. Wait for processing  
3. View predicted emotion & confidence
    """
)

st.sidebar.markdown("### Emotion Classes")
st.sidebar.write(", ".join(label_encoder.classes_))

st.sidebar.markdown("---")
st.sidebar.caption("Built as a Machine Learning Portfolio Project")

# =============================
# Main UI — ONLINE INFERENCE
# =============================
st.title("🎧 Speech Emotion Recognition Dashboard")
st.write("Upload a speech `.wav` file to analyze its emotional tone.")

uploaded_file = st.file_uploader("Choose a `.wav` file", type=["wav"])

if uploaded_file is not None:
    col_left, col_right = st.columns([1.1, 1.3])

    with col_left:
        st.subheader("🔊 Audio Preview")
        st.audio(uploaded_file, format="audio/wav")

    with st.spinner("Analyzing audio..."):
        try:
            # Process audio
            y, sr, duration, mfcc_mean, mel_db = process_audio(uploaded_file)

            # Scale & predict
            features_scaled = scaler.transform([mfcc_mean])
            proba = model.predict_proba(features_scaled)[0]
            pred_class = np.argmax(proba)
            emotion_label = label_encoder.inverse_transform([pred_class])[0]
            confidence = proba[pred_class]

            with col_left:
                st.subheader("🧠 Prediction")
                st.metric("Predicted Emotion", emotion_label.upper())
                st.metric("Model Confidence", f"{confidence * 100:.1f}%")
                st.metric("Audio Duration", f"{duration:.2f} s")

                st.subheader("📊 Emotion Probabilities")
                proba_df = pd.DataFrame(
                    {"Emotion": label_encoder.classes_, "Probability": proba}
                ).set_index("Emotion")
                st.bar_chart(proba_df)

            with col_right:
                st.subheader("📈 Waveform")
                fig_wav, ax_wav = plt.subplots(figsize=(8, 2.5))
                librosa.display.waveshow(y, sr=sr, ax=ax_wav)
                ax_wav.set_xlabel("Time (s)")
                ax_wav.set_ylabel("Amplitude")
                ax_wav.set_title("Audio Waveform")
                st.pyplot(fig_wav)

                st.subheader("🌈 Mel-Spectrogram")
                fig_mel, ax_mel = plt.subplots(figsize=(8, 3.5))
                img = librosa.display.specshow(
                    mel_db,
                    x_axis="time",
                    y_axis="mel",
                    sr=sr,
                    ax=ax_mel
                )
                fig_mel.colorbar(img, ax=ax_mel, format="%+2.0f dB")
                ax_mel.set_title("Mel-Spectrogram (dB)")
                st.pyplot(fig_mel)

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

else:
    st.info("Please upload a `.wav` file to begin.")

# ======================================================
# OFFLINE MODEL EVALUATION (PHASE 1 — ALWAYS VISIBLE)
# ======================================================
st.markdown("---")
st.header("📊 Model Evaluation (Offline Test Set)")

st.metric(
    label="Model Accuracy",
    value=f"{eval_accuracy * 100:.2f}%"
)

st.subheader("Confusion Matrix")

fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.5))

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    ax=ax_cm
)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ax_cm.set_title("Speech Emotion Recognition – Confusion Matrix")

st.pyplot(fig_cm)
