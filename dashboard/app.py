"""
Professional Health Insurance Anomaly Detection Dashboard
For Nigerian HMO Claims Auditors

Features:
- File upload with validation
- Model selection (Ensemble + 4 individual models)
- Interactive anomaly ranking
- Professional visualizations
- SHAP explanations for flagged claims
- Export functionality
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Fraud Detection System | Nigerian HMOs",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #1a365d 0%, #2b6cb0 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header-title {
        color: white !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .header-subtitle {
        color: rgba(255,255,255,0.85) !important;
        font-size: 1.1rem !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .header-badge {
        background: rgba(255,255,255,0.15);
        padding: 0.3rem 1rem;
        border-radius: 20px;
        color: white;
        font-size: 0.85rem;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid #2b6cb0;
        transition: transform 0.2s;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a365d;
        margin: 0;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #718096;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Cards */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.75rem;
    }
    
    /* Status badges */
    .badge-success {
        background: #48bb78;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
    }
    .badge-danger {
        background: #fc8181;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
    }
    .badge-warning {
        background: #ecc94b;
        color: #1a202c;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
    }
    
    /* Data table */
    .dataframe {
        font-size: 0.85rem !important;
    }
    .dataframe thead th {
        background: #f7fafc !important;
        color: #2d3748 !important;
        font-weight: 600 !important;
        padding: 0.75rem 0.5rem !important;
    }
    .dataframe tbody td {
        padding: 0.5rem 0.5rem !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #a0aec0;
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #e2e8f0;
        margin-top: 2rem;
    }
    
    /* Sidebar */
    .sidebar-content {
        padding: 0.5rem 0;
    }
    .sidebar-section {
        margin-bottom: 1.5rem;
    }
    .sidebar-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    
    /* Buttons */
    .stButton button {
        width: 100%;
        background: #2b6cb0;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        transition: background 0.2s;
    }
    .stButton button:hover {
        background: #1a365d;
        color: white;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: #2d3748 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        color: #4a5568;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #2b6cb0;
        border-bottom-color: #2b6cb0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================

st.markdown("""
<div class="header-container">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h2 class="header-title">🏥 Health Insurance Fraud Detection</h2>
            <p class="header-subtitle">Unsupervised Anomaly Detection System for Nigerian HMOs</p>
            <span class="header-badge">⚡ AI-Powered • Real-Time • Explainable</span>
        </div>
        <!--<div style="text-align: right;">
            <span style="color: rgba(255,255,255,0.7); font-size: 0.8rem;">Powered by</span>
            <br>
            <span style="color: white; font-weight: 600;">Streamlit • Python • SHAP</span>
        </div>-->
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload Claims Data",
        type=['csv'],
        help="Upload a CSV file with claims data."
    )
    
    st.markdown("---")
    
    # Model selection
    st.markdown("#### 🤖 Detection Model")
    model_choice = st.selectbox(
        "Select Model",
        ["Ensemble", "Isolation Forest", "One-Class SVM", "Local Outlier Factor", "Autoencoder"],
        help="Ensemble combines all four models for best performance."
    )
    
    st.markdown("---")
    
    # Parameters
    st.markdown("#### 📊 Display Options")
    top_k = st.slider(
        "Number of Anomalies",
        min_value=10,
        max_value=500,
        value=100,
        step=10
    )
    
    st.markdown("---")
    
    # Help section
    st.markdown("#### ℹ️ How It Works")
    st.markdown("""
    1. **Upload** your claims data
    2. **Select** a detection model
    3. **Review** ranked anomalies
    4. **Investigate** flagged claims
    
    **Models:**
    - **Ensemble** (Recommended) - Combines all models
    - **Isolation Forest** - Fast & robust
    - **One-Class SVM** - Complex boundaries
    - **LOF** - Local density analysis
    - **Autoencoder** - Neural network detection
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.7rem; color: #a0aec0; text-align: center;">
        Built for Nigerian HMOs
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# LOAD MODELS
# =============================================================================

@st.cache_resource
def load_models():
    """Load trained models and preprocessing objects."""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_root, 'models')
        data_dir = os.path.join(project_root, 'data', 'processed')
        
        models = {}
        
        # Load scikit-learn models
        with open(os.path.join(models_dir, 'isolation_forest.pkl'), 'rb') as f:
            models['if'] = pickle.load(f)
        
        with open(os.path.join(models_dir, 'one_class_svm.pkl'), 'rb') as f:
            models['ocsvm'] = pickle.load(f)
        
        with open(os.path.join(models_dir, 'lof.pkl'), 'rb') as f:
            models['lof'] = pickle.load(f)
        
        # Load autoencoder
        from keras import models as keras_models
        models['ae'] = keras_models.load_model(
            os.path.join(models_dir, 'autoencoder.h5'),
            compile=False
        )
        
        # Load scaler and feature names
        with open(os.path.join(data_dir, 'scaler.pkl'), 'rb') as f:
            models['scaler'] = pickle.load(f)
        
        with open(os.path.join(data_dir, 'feature_names.txt'), 'r') as f:
            models['feature_names'] = f.read().splitlines()
        
        return models
    
    except Exception as e:
        st.error(f"⚠️ Error loading models: {str(e)}")
        st.info("Please run the model training first.")
        return None


def get_scores(model, X):
    """Get anomaly scores from any model."""
    try:
        if hasattr(model, 'score_samples'):
            return -model.score_samples(X)
        elif hasattr(model, 'decision_function'):
            return -model.decision_function(X)
        elif hasattr(model, 'predict'):
            pred = model.predict(X)
            if pred.min() >= -1 and pred.max() <= 1:
                return -pred.flatten()
            return pred.flatten()
        else:
            return None
    except:
        return None


def engineer_features(df, feature_names, scaler):
    """Engineer features matching training."""
    df2 = df.copy()
    
    # Date features
    if 'date_of_service' in df2.columns:
        df2['date_of_service'] = pd.to_datetime(df2['date_of_service'])
        df2['year'] = df2['date_of_service'].dt.year
        df2['month'] = df2['date_of_service'].dt.month
        df2['day_of_week'] = df2['date_of_service'].dt.dayofweek
        df2['day_of_month'] = df2['date_of_service'].dt.day
        df2['quarter'] = df2['date_of_service'].dt.quarter
    
    # Frequency encoding
    if 'provider_npi' in df2.columns:
        freq = df2['provider_npi'].value_counts(normalize=True)
        df2['provider_freq'] = df2['provider_npi'].map(freq)
    
    if 'cpt_procedure_code' in df2.columns:
        freq = df2['cpt_procedure_code'].value_counts(normalize=True)
        df2['procedure_freq'] = df2['cpt_procedure_code'].map(freq)
    
    # Provider aggregates
    if 'provider_npi' in df2.columns:
        agg = df2.groupby('provider_npi')['billed_amount'].agg(['mean', 'std', 'count']).reset_index()
        agg.columns = ['provider_npi', 'provider_mean', 'provider_std', 'provider_volume']
        df2 = df2.merge(agg, on='provider_npi', how='left')
    
    # Rolling averages
    if 'date_of_service' in df2.columns and 'provider_npi' in df2.columns:
        df2 = df2.sort_values(['provider_npi', 'date_of_service'])
        df2['rolling_mean_30d'] = df2.groupby('provider_npi')['billed_amount'].transform(
            lambda x: x.rolling(30, min_periods=1).mean()
        )
        df2['rolling_std_30d'] = df2.groupby('provider_npi')['billed_amount'].transform(
            lambda x: x.rolling(30, min_periods=1).std()
        )
    
    # Ensure all features exist
    for col in feature_names:
        if col not in df2.columns:
            df2[col] = 0.0
    
    # Build matrix
    X = df2[feature_names].copy()
    X = X.fillna(X.mean())
    
    # Scale
    X_scaled = scaler.transform(X)
    
    return X_scaled, df2


def normalise(scores):
    if scores is None:
        return None
    if scores.max() == scores.min():
        return scores
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main dashboard application."""
    
    # Load models
    models = load_models()
    
    if models is None:
        st.stop()
    
    # =========================================================================
    # WELCOME SCREEN
    # =========================================================================
    
    if uploaded_file is None:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
            <h2 style="color: #1a365d; margin-bottom: 0.5rem;">Ready to Detect Fraud</h2>
            <p style="color: #4a5568; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
                Upload your claims data to start detecting suspicious patterns using 
                advanced unsupervised machine learning.
            </p>
            <div style="margin-top: 2rem; display: flex; gap: 1rem; justify-content: center;">
                <div style="background: #f7fafc; padding: 1rem 1.5rem; border-radius: 8px;">
                    <span style="font-size: 1.5rem;">📁</span>
                    <p style="margin: 0; color: #2d3748; font-weight: 600;">Upload CSV</p>
                </div>
                <div style="background: #f7fafc; padding: 1rem 1.5rem; border-radius: 8px;">
                    <span style="font-size: 1.5rem;">🤖</span>
                    <p style="margin: 0; color: #2d3748; font-weight: 600;">AI Detection</p>
                </div>
                <div style="background: #f7fafc; padding: 1rem 1.5rem; border-radius: 8px;">
                    <span style="font-size: 1.5rem;">💡</span>
                    <p style="margin: 0; color: #2d3748; font-weight: 600;">SHAP Explain</p>
                </div>
            </div>
            <div style="margin-top: 2rem; background: #ebf8ff; padding: 1rem; border-radius: 8px; max-width: 500px; margin-left: auto; margin-right: auto;">
                <p style="color: #2b6cb0; margin: 0; font-size: 0.9rem;">
                    <strong>💡 Tip:</strong> Use the sidebar to upload your claims data
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
      
        # Sample data button
        #col1, col2, col3 = st.columns([1, 1, 1])
    
        #with col2:
         #   if st.button("🚀 Try with Sample Data", use_container_width=False):
          #      st.session_state['use_sample'] = True
           #     st.rerun()
        #return
    
    
    # =========================================================================
    # PROCESS UPLOADED DATA
    # =========================================================================
    
    try:
        # Read data
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        elif st.session_state.get('use_sample', False):
            # Generate sample data
            np.random.seed(42)
            n = 1000
            df = pd.DataFrame({
                'claim_id': [f'CLM_{i:06d}' for i in range(n)],
                'provider_npi': np.random.choice([f'PROV_{i:03d}' for i in range(20)], n),
                'patient_id': np.random.choice([f'PAT_{i:04d}' for i in range(200)], n),
                'billed_amount': np.random.lognormal(10, 1.0, n),
                'date_of_service': pd.date_range('2024-01-01', periods=n, freq='D'),
                'cpt_procedure_code': np.random.choice([f'CPT_{i:04d}' for i in range(10)], n),
                'claim_status': np.random.choice(['Approved', 'Pending', 'Denied', 'Under Review'], n)
            })
            # Inject anomalies
            anomaly_idx = np.random.choice(n, 50, replace=False)
            df.loc[anomaly_idx, 'billed_amount'] *= np.random.uniform(3, 5, 50)
            st.session_state['use_sample'] = False
        else:
            #st.warning("Please upload a CSV file.")
            return
        
        # Validate columns
        required_cols = ['claim_id', 'provider_npi', 'patient_id', 'billed_amount']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.info("Required columns: claim_id, provider_npi, patient_id, billed_amount")
            return
        
        # =====================================================================
        # PROCESS
        # =====================================================================
        
        with st.spinner("🔍 Detecting anomalies..."):
            # Engineer features
            X_scaled, df_processed = engineer_features(
                df, 
                models['feature_names'], 
                models['scaler']
            )
            
            # Get model scores
            all_scores = {}
            
            # Isolation Forest
            s = get_scores(models['if'], X_scaled)
            if s is not None:
                all_scores['Isolation Forest'] = normalise(s)
            
            # One-Class SVM
            s = get_scores(models['ocsvm'], X_scaled)
            if s is not None:
                all_scores['One-Class SVM'] = normalise(s)
            
            # LOF
            s = get_scores(models['lof'], X_scaled)
            if s is not None:
                all_scores['Local Outlier Factor'] = normalise(s)
            
            # Autoencoder
            try:
                rec = models['ae'].predict(X_scaled, verbose=0)
                ae_s = np.mean((X_scaled - rec)**2, axis=1)
                all_scores['Autoencoder'] = normalise(ae_s)
            except:
                pass
            
            # Ensemble
            if all_scores:
                ensemble = np.mean(list(all_scores.values()), axis=0)
                all_scores['Ensemble'] = normalise(ensemble)
            else:
                st.error("No models produced scores. Please check your models.")
                return
            
            # Select scores
            if model_choice in all_scores:
                final_scores = all_scores[model_choice]
            else:
                final_scores = all_scores.get('Ensemble', np.zeros(X_scaled.shape[0]))
            
            # Add to dataframe
            df_processed['anomaly_score'] = final_scores
            threshold = np.percentile(final_scores, 95)
            df_processed['is_anomaly'] = final_scores > threshold
            df_processed['risk_level'] = pd.cut(
                final_scores,
                bins=[0, 0.3, 0.7, 1.0],
                labels=['Low', 'Medium', 'High']
            )
            
            # Rank
            df_ranked = df_processed.sort_values('anomaly_score', ascending=False)
            df_ranked['rank'] = range(1, len(df_ranked) + 1)
            top_anomalies = df_ranked.head(top_k)
        
        # =====================================================================
        # DISPLAY METRICS
        # =====================================================================
        
        st.markdown("### 📊 Dashboard Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Total Claims</p>
                <p class="metric-value">{len(df_ranked):,}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #48bb78;">
                <p class="metric-label">Avg Anomaly Score</p>
                <p class="metric-value">{final_scores.mean():.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #ecc94b;">
                <p class="metric-label">Max Risk Score</p>
                <p class="metric-value">{final_scores.max():.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            high_risk = (df_ranked['risk_level'] == 'High').sum()
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #fc8181;">
                <p class="metric-label">High Risk Claims</p>
                <p class="metric-value">{high_risk}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            flagged = top_anomalies['is_anomaly'].sum()
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #9f7aea;">
                <p class="metric-label">Flagged Anomalies</p>
                <p class="metric-value">{flagged}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # =====================================================================
        # RESULTS TABS
        # =====================================================================
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Anomalies Table",
            "📈 Visualizations",
            "💡 SHAP Explanations",
            "📋 Data Preview"
        ])
        
        # =====================================================================
        # TAB 1: ANOMALIES TABLE
        # =====================================================================
        
        with tab1:
            st.markdown(f"### Top {top_k} Suspicious Claims")
            
            display_cols = ['rank', 'claim_id', 'provider_npi', 'patient_id', 'billed_amount', 'anomaly_score', 'risk_level']
            for col in ['cpt_procedure_code', 'date_of_service', 'claim_status']:
                if col in top_anomalies.columns:
                    display_cols.append(col)
            
            st.dataframe(
                top_anomalies[display_cols].style.format({
                    'billed_amount': '₦{:.2f}',
                    'anomaly_score': '{:.4f}'
                }).background_gradient(
                    subset=['anomaly_score'],
                    cmap='RdYlGn_r',
                    vmin=0,
                    vmax=1
                ),
                use_container_width=True,
                height=500
            )
        
        # =====================================================================
        # TAB 2: VISUALIZATIONS
        # =====================================================================
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Score distribution
                fig, ax = plt.subplots(figsize=(8, 5))
                scores = df_ranked['anomaly_score']
                threshold = np.percentile(scores, 95)
                
                ax.hist(scores, bins=50, alpha=0.7, color='#2b6cb0', edgecolor='black')
                ax.axvline(threshold, color='#e53e3e', linestyle='--', linewidth=2, 
                          label=f'95th percentile: {threshold:.3f}')
                ax.axvline(scores.mean(), color='#38a169', linestyle='-', linewidth=2,
                          label=f'Mean: {scores.mean():.3f}')
                ax.set_xlabel('Anomaly Score', fontsize=11)
                ax.set_ylabel('Frequency', fontsize=11)
                ax.set_title('Anomaly Score Distribution', fontsize=13, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                # Provider risk heatmap
                provider_risk = df_ranked.groupby('provider_npi')['anomaly_score'].mean().sort_values(ascending=False).head(15)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                # FIX: use sns.color_palette instead of plt.cm
                colors = sns.color_palette("RdYlGn_r", len(provider_risk))
                provider_risk.plot(kind='barh', ax=ax, color=colors)
                ax.set_xlabel('Average Anomaly Score', fontsize=11)
                ax.set_ylabel('Provider NPI', fontsize=11)
                ax.set_title('Top 15 Providers by Average Risk', fontsize=13, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                plt.tight_layout()
                st.pyplot(fig)
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Risk level distribution
                risk_counts = df_ranked['risk_level'].value_counts()
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = {'Low': '#48bb78', 'Medium': '#ecc94b', 'High': '#fc8181'}
                risk_counts.plot(kind='bar', ax=ax, color=[colors.get(x, '#a0aec0') for x in risk_counts.index])
                ax.set_xlabel('Risk Level', fontsize=11)
                ax.set_ylabel('Count', fontsize=11)
                ax.set_title('Risk Level Distribution', fontsize=13, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col4:
                # Score correlation
                score_df = pd.DataFrame({
                    'IF': all_scores.get('Isolation Forest', np.zeros_like(final_scores)),
                    'SVM': all_scores.get('One-Class SVM', np.zeros_like(final_scores)),
                    'LOF': all_scores.get('Local Outlier Factor', np.zeros_like(final_scores)),
                    'AE': all_scores.get('Autoencoder', np.zeros_like(final_scores))
                })
                corr = score_df.corr()
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm',
                           square=True, cbar_kws={"shrink": 0.8}, ax=ax)
                ax.set_title('Model Score Correlation', fontsize=13, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
        
        # =====================================================================
        # TAB 3: SHAP EXPLANATIONS
        # =====================================================================
        
        with tab3:
            st.markdown("### 💡 Explainable AI (SHAP)")
            st.markdown("Select a claim to understand why it was flagged as suspicious.")
            
            top_5 = top_anomalies.head(5)
            
            if len(top_5) > 0:
                options = [
                    f"Rank {row['rank']}: {row['claim_id']} (Score: {row['anomaly_score']:.3f})"
                    for _, row in top_5.iterrows()
                ]
                
                selected = st.selectbox("Select Claim", options)
                
                if selected:
                    idx = options.index(selected)
                    claim = top_5.iloc[idx]
                    
                    # Display claim details
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"**Provider:** {claim['provider_npi']}")
                    with col2:
                        st.info(f"**Patient:** {claim['patient_id']}")
                    with col3:
                        st.info(f"**Amount:** ₦{claim['billed_amount']:,.2f}")
                    
                    st.info(f"**Risk Score:** {claim['anomaly_score']:.4f} | **Risk Level:** {claim['risk_level']}")
                    
                    try:
                        # SHAP explanation
                        explainer = shap.KernelExplainer(models['if'].predict, X_scaled[:100])
                        shap_values = explainer.shap_values(X_scaled[claim.name].reshape(1, -1))
                        
                        # Force plot
                        fig = plt.figure(figsize=(12, 3))
                        shap.force_plot(
                            explainer.expected_value,
                            shap_values[0],
                            X_scaled[claim.name],
                            feature_names=models['feature_names'],
                            matplotlib=True,
                            show=False
                        )
                        plt.title(f'SHAP Force Plot - {claim["claim_id"]}', fontsize=12)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Feature contributions
                        contributions = pd.DataFrame({
                            'Feature': models['feature_names'],
                            'SHAP Value': shap_values[0]
                        }).sort_values('SHAP Value', ascending=False)
                        
                        st.markdown("**Feature Contributions:**")
                        st.dataframe(
                            contributions.style.format({'SHAP Value': '{:.4f}'})
                            .background_gradient(subset=['SHAP Value'], cmap='RdBu_r'),
                            use_container_width=True
                        )
                    except Exception as e:
                        st.warning(f"SHAP explanation unavailable: {str(e)}")
            else:
                st.info("No anomalies to explain.")
        
        # =====================================================================
        # TAB 4: DATA PREVIEW
        # =====================================================================
        
        with tab4:
            st.markdown("### 📋 Data Preview")
            st.dataframe(df.head(20), use_container_width=True)
            
            st.markdown("### 📊 Data Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Numeric Columns:**")
                st.dataframe(df.describe(), use_container_width=True)
            
            with col2:
                st.markdown("**Column Info:**")
                info_df = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Nulls': df.isnull().sum().values,
                    'Unique': df.nunique().values
                })
                st.dataframe(info_df, use_container_width=True)
        
        # =====================================================================
        # DOWNLOAD
        # =====================================================================
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.download_button(
                label="📥 Download Flagged Claims CSV",
                data=top_anomalies.to_csv(index=False).encode('utf-8'),
                file_name=f'flagged_claims_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv',
                use_container_width=True
            )
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        with st.expander("📖 Show Error Details"):
            st.write(e)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("""
<div class="footer">
    <p>🏥 Health Insurance Anomaly Detection System &copy; 2026</p>
    <!--<p>Built for Nigerian Health Maintenance Organisations (HMOs) • MSc Project</p>-->
    <p>Powered by Houzibe Denis</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    main()