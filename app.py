"""
MalwareVision - Deep Learning Malware Detection System
======================================================
Student: Mohit Gaur | Roll Number: 24UITE5030
B.Tech 7th Semester Minor Project

A modern Streamlit application providing:
1. Malware Scanner: Upload sample images or select curated held-out test samples.
2. Model Performance: Actual empirical metrics, confusion matrix, and training curves.
3. About Project: Complete system architecture, dataset methodology, and viva documentation.
"""

import os
import json
from pathlib import Path
from PIL import Image
import streamlit as st

# Configure Streamlit Page
st.set_page_config(
    page_title="MalwareVision | Minor Project",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Academic Presentation
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #1E40AF;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-malicious {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #F87171;
        border-radius: 8px;
        padding: 1rem;
        font-weight: 700;
        font-size: 1.2rem;
        text-align: center;
    }
    .badge-benign {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #4ADE80;
        border-radius: 8px;
        padding: 1rem;
        font-weight: 700;
        font-size: 1.2rem;
        text-align: center;
    }
    .disclaimer-box {
        background-color: #F1F5F9;
        border-left: 4px solid #64748B;
        padding: 0.8rem 1rem;
        font-size: 0.88rem;
        color: #475569;
        border-radius: 4px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load prediction pipeline safely
@st.cache_resource
def load_predict_engine():
    try:
        from src.predict import predict, get_model
        # Pre-warm model load
        get_model()
        return predict, None
    except Exception as e:
        return None, str(e)


# Helper function to load results
def load_saved_results():
    metrics_path = "results/metrics.json"
    history_path = "results/training_history.json"
    summary_path = "dataset/dataset_summary.json"
    
    metrics = {}
    history = {}
    summary = {}
    
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
            
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
            
    return metrics, history, summary


# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shield.png", width=64)
    st.markdown("### **MalwareVision**")
    st.caption("CNN-Based Binary Malware Image Detection")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🔍 Malware Scanner", "📊 Model Performance", "ℹ️ About Project"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("**Academic Information**")
    st.caption("**Student:** Mohit Gaur")
    st.caption("**Roll Number:** 24UITE5030")
    st.caption("**Degree:** B.Tech 7th Semester")
    st.caption("**Course:** Minor Project")


# ==============================================================================
# PAGE 1: MALWARE SCANNER
# ==============================================================================
if page == "🔍 Malware Scanner":
    st.markdown('<div class="main-title">🛡️ MalwareVision Scanner</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload a binary grayscale sample image or select a held-out test sample to detect malware patterns.</div>', unsafe_allow_html=True)
    
    predict_fn, err_msg = load_predict_engine()
    
    if err_msg:
        st.error(f"⚠️ Model Initialization Notice: {err_msg}")
        st.info("💡 To train the model and generate weights, run: `python src/train.py` from the project directory.")
        st.stop()

    col1, col2 = st.columns([1, 1], gap="large")
    
    selected_image = None
    sample_origin = ""

    with col1:
        st.markdown("#### **Input Sample Image**")
        input_choice = st.radio("Choose Input Method:", ["Upload Image File", "Select Pre-Loaded Demo Sample"], horizontal=True)
        
        if input_choice == "Upload Image File":
            uploaded_file = st.file_uploader(
                "Upload a binary visualization image (PNG, JPG, JPEG):",
                type=["png", "jpg", "jpeg"],
                help="Accepts grayscale or RGB byte-visualization images."
            )
            if uploaded_file is not None:
                try:
                    selected_image = Image.open(uploaded_file)
                    sample_origin = f"Uploaded File: {uploaded_file.name}"
                except Exception as ex:
                    st.error(f"Invalid or corrupted image file: {ex}")
        else:
            st.markdown("##### **Select Curated Demo Sample (Held-out from training):**")
            sample_type = st.selectbox("Sample Category:", ["Malicious Test Samples", "Benign (Clean) Test Samples"])
            sample_folder = "samples/malicious" if "Malicious" in sample_type else "samples/benign"
            
            if os.path.exists(sample_folder):
                samples_list = [f for f in os.listdir(sample_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if samples_list:
                    selected_sample_file = st.selectbox("Choose Sample:", samples_list, format_func=lambda x: f"Sample: {x[:28]}...")
                    full_sample_path = os.path.join(sample_folder, selected_sample_file)
                    selected_image = Image.open(full_sample_path)
                    sample_origin = f"Demo Sample: {selected_sample_file}"
                else:
                    st.warning("No sample files found in the demo samples directory.")
            else:
                st.warning("Samples directory not initialized.")

        if selected_image is not None:
            st.markdown("---")
            st.caption(f"**Loaded:** {sample_origin}")
            st.image(selected_image, caption=f"Original Dimensions: {selected_image.size[0]}×{selected_image.size[1]} | Mode: {selected_image.mode}", use_container_width=True)

    with col2:
        st.markdown("#### **Detection & Analysis**")
        if selected_image is None:
            st.info("👈 Upload an image or select a demo sample to begin analysis.")
        else:
            analyze_btn = st.button("🚀 Analyze Sample Image", type="primary", use_container_width=True)
            
            if analyze_btn:
                with st.spinner("Processing grayscale normalization and running CNN inference..."):
                    result = predict_fn(selected_image)
                    
                st.markdown("---")
                
                is_mal = result["prediction"] == "Malicious"
                conf_pct = result["confidence"] * 100
                mal_prob = result["malware_probability"] * 100
                
                if is_mal:
                    st.markdown(
                        f"""
                        <div class="badge-malicious">
                            🚨 Malicious Pattern Detected<br>
                            <span style="font-size: 0.95rem; font-weight: 500;">Structural signatures consistent with malware binaries</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="badge-benign">
                            ✅ Benign (Safe) Pattern Predicted<br>
                            <span style="font-size: 0.95rem; font-weight: 500;">Structural signatures consistent with legitimate executables</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("###")
                mcol1, mcol2 = st.columns(2)
                with mcol1:
                    st.metric("Model Confidence", f"{conf_pct:.2f}%")
                with mcol2:
                    st.metric("Malware Probability", f"{mal_prob:.2f}%")

                st.progress(result["confidence"])
                
                with st.expander("🔍 View Preprocessed 128×128 Tensor Preview"):
                    from src.preprocessing import preprocess_image
                    proc = preprocess_image(selected_image)
                    st.write(f"Tensor Shape: `{proc.shape}` | Dtype: `{proc.dtype}` | Min: `{proc.min():.3f}` | Max: `{proc.max():.3f}`")
                    # Squeeze back to 2D for preview
                    st.image(proc.squeeze(), caption="Standardized 128×128 Grayscale Input to CNN", width=200)

                st.markdown("""
                <div class="disclaimer-box">
                    ℹ️ <b>Academic Prototype Note:</b> This system uses convolutional spatial texture recognition on byte-mapped grayscale images. It does not execute binary code or replace active anti-virus engines.
                </div>
                """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 2: MODEL PERFORMANCE DASHBOARD
# ==============================================================================
elif page == "📊 Model Performance":
    st.markdown('<div class="main-title">📊 Empirical Model Performance Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Evaluation results calculated on 600 unseen test samples (300 benign, 300 malicious) with 70:15:15 stratified partitioning.</div>', unsafe_allow_html=True)
    
    metrics, history, summary = load_saved_results()
    
    if not metrics:
        st.warning("⚠️ Evaluation metrics have not been generated yet. Please run `python src/evaluate.py` to produce results.")
    else:
        # Key Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('accuracy', 0)*100:.2f}%</div>
                <div class="metric-label">Test Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('precision', 0)*100:.2f}%</div>
                <div class="metric-label">Precision</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('recall', 0)*100:.2f}%</div>
                <div class="metric-label">Recall (Sensitivity)</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('f1_score', 0)*100:.2f}%</div>
                <div class="metric-label">F1-Score</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("###")
        
        # Dataset Summary Table
        dcol1, dcol2 = st.columns([1, 1], gap="large")
        with dcol1:
            st.markdown("#### **Dataset Partitioning (Zero Data Leakage)**")
            st.markdown("""
            | Split Partition | Benign Samples | Malicious Samples | Total Samples | Percentage |
            |---|---|---|---|---|
            | **Training Set** | 1,400 | 1,400 | 2,800 | **70%** |
            | **Validation Set** | 300 | 300 | 600 | **15%** |
            | **Testing Set (Held-out)** | 300 | 300 | 600 | **15%** |
            | **Total Research Corpus** | **2,000** | **2,000** | **4,000** | **100%** |
            """)
            st.caption("🔒 Deduplicated using SHA-256 cryptographic hashes before stratified splitting.")

        with dcol2:
            st.markdown("#### **Test Classification Breakdown**")
            cm_dict = metrics.get("confusion_matrix", {})
            tn = cm_dict.get("true_negatives", 0)
            fp = cm_dict.get("false_positives", 0)
            fn = cm_dict.get("false_negatives", 0)
            tp = cm_dict.get("true_positives", 0)
            total_test = tn + fp + fn + tp
            
            st.markdown(f"""
            - **True Negatives (Correct Benign):** `{tn}` / 300 ({tn/300*100:.2f}%)
            - **True Positives (Correct Malware):** `{tp}` / 300 ({tp/300*100:.2f}%)
            - **False Positives (Benign misclassified as Malware):** `{fp}` / 300 ({fp/300*100:.2f}%)
            - **False Negatives (Malware misclassified as Benign):** `{fn}` / 300 ({fn/300*100:.2f}%)
            - **Total Evaluated Unseen Samples:** `{total_test}`
            """)

        st.markdown("---")
        
        # Visualizations
        gcol1, gcol2 = st.columns([1, 1], gap="large")
        
        with gcol1:
            st.markdown("#### **Confusion Matrix Heatmap**")
            cm_img_path = "results/confusion_matrix.png"
            if os.path.exists(cm_img_path):
                st.image(cm_img_path, caption="Confusion Matrix on 600 Unseen Test Samples", use_container_width=True)
            else:
                st.info("Confusion matrix graph not found.")

        with gcol2:
            st.markdown("#### **Training & Validation Accuracy**")
            acc_img_path = "results/accuracy_graph.png"
            if os.path.exists(acc_img_path):
                st.image(acc_img_path, caption="Training vs Validation Accuracy across Epochs", use_container_width=True)
            else:
                st.info("Accuracy graph not found.")

        st.markdown("###")
        st.markdown("#### **Training & Validation Loss Curve**")
        loss_img_path = "results/loss_graph.png"
        if os.path.exists(loss_img_path):
            st.image(loss_img_path, caption="Binary Cross-Entropy Loss Curve showing convergence without severe overfitting", use_container_width=True)

        if history:
            with st.expander("📋 View Complete Epoch-by-Epoch Numerical Log"):
                import pandas as pd
                df_hist = pd.DataFrame({
                    "Epoch": history.get("epochs", []),
                    "Training Accuracy": [f"{v*100:.2f}%" for v in history.get("accuracy", [])],
                    "Validation Accuracy": [f"{v*100:.2f}%" for v in history.get("val_accuracy", [])],
                    "Training Loss": [f"{v:.4f}" for v in history.get("loss", [])],
                    "Validation Loss": [f"{v:.4f}" for v in history.get("val_loss", [])]
                })
                st.dataframe(df_hist, use_container_width=True, hide_index=True)


# ==============================================================================
# PAGE 3: ABOUT PROJECT & VIVA DOCUMENTATION
# ==============================================================================
elif page == "ℹ️ About Project":
    st.markdown('<div class="main-title">ℹ️ About MalwareVision</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">B.Tech 7th Semester Minor Project Overview, Architecture, and Technical Documentation.</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 👨‍🎓 Project Credits & Details
    - **Student Name:** Mohit Gaur
    - **Roll Number:** 24UITE5030
    - **Project Level:** B.Tech 7th Semester Minor Project
    - **Estimated Development Time:** 5–7 Days
    - **Domain:** Cybersecurity & Deep Learning (Computer Vision)
    
    ---

    ### 🎯 Problem Statement & Background
    Traditional malware detection relies primarily on **signature-based hashing** (MD5/SHA256) and heuristic static analysis. However:
    1. Attackers bypass signatures easily using **polymorphism, metamorphism, and packing**.
    2. Dynamic sandbox analysis requires executing suspicious binaries in virtual machines, which is resource-intensive and vulnerable to anti-sandbox detection tricks.
    
    **The Deep Learning Solution:**  
    Mapping binary executables to **grayscale image matrices** (Nataraj et al. 2011) transforms malware detection into a spatial texture classification task. By assigning each raw byte $[0, 255]$ to a corresponding pixel intensity, distinct file components (headers, .text code sections, resource tables, and encrypted payloads) produce characteristic visual patterns. A Convolutional Neural Network (CNN) learns these visual signatures directly **without executing the binary**.

    ---

    ### 🏗️ End-to-End System Architecture

    ```
    [Raw Grayscale Sample]
              │
              ▼
    [Image Preprocessing Pipeline] ──► Resize 128×128, 8-bit Grayscale, Pixel Normalization [0.0, 1.0]
              │
              ▼
    [Custom 3-Block CNN]
        ├── Block 1: Conv2D(32, 3×3, ReLU) + MaxPooling2D(2×2)
        ├── Block 2: Conv2D(64, 3×3, ReLU) + MaxPooling2D(2×2)
        ├── Block 3: Conv2D(128, 3×3, ReLU) + MaxPooling2D(2×2)
        ├── GlobalAveragePooling2D()
        ├── Dense(64, ReLU) + Dropout(0.3)
        └── Output: Dense(1, Sigmoid)
              │
              ▼
    [Prediction Head] ──► Classification (Benign vs Malicious) + Confidence Score
              │
              ▼
    [Streamlit Web Dashboard] ──► Scanner, Metric Dashboard, Viva Presentation
    ```

    ---

    ### 🛠️ Technology Stack
    - **Programming Language:** Python 3.11
    - **Deep Learning Framework:** TensorFlow 2.x & Keras
    - **Image Processing:** Pillow (PIL)
    - **Scientific Computation:** NumPy, Pandas, Scikit-learn
    - **Visualization:** Matplotlib, Seaborn
    - **Web Framework:** Streamlit
    - **Model Storage:** Native Keras (`.keras`) format

    ---

    ### ⚖️ Project Scope (V1 vs Future Scope)

    | Built in V1 (This Minor Project) | Explicitly Excluded (Future Scope) |
    |---|---|
    | ✅ Prepared grayscale image dataset (4,000 samples) | ❌ Real-time live file system monitoring |
    | ✅ SHA-256 deduplication & zero-leakage split | ❌ Live binary execution or quarantine engine |
    | ✅ 3-block CNN trained with EarlyStopping | ❌ Multi-family fine-grained malware classification |
    | ✅ Full evaluation metrics on 600 unseen test samples | ❌ User authentication & cloud databases |
    | ✅ Interactive Streamlit web interface | ❌ Paid third-party antivirus API integrations |

    ---

    ### 💡 Viva Presentation Key Talking Points
    1. **Why Grayscale Representation?**  
       Byte values range from $0$ (`0x00`) to $255$ (`0xFF`), which maps perfectly to 8-bit grayscale pixel intensities without requiring arbitrary encodings.
    2. **Why GlobalAveragePooling instead of Flatten?**  
       Global Average Pooling drastically reduces model parameter count (from millions to ~101K), minimizing overfitting and enabling fast CPU training.
    3. **How was Data Leakage Prevented?**  
       Cryptographic SHA-256 hashes were calculated across all raw image files to identify and eliminate duplicates before applying stratified 70:15:15 partitioning.
    4. **How are Thresholds Interpreted?**  
       Sigmoid output $\ge 0.5$ classifies as Malicious; $< 0.5$ classifies as Benign. The confidence score reflects the distance from the decision boundary.
    """)
