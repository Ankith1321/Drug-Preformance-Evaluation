# app.py
# -----------------------------------------------------------------------------
# Drug Performance Analytics Suite
# (Interactive EDA, Classification, Clustering, and Regression)
#
# Core modeling + EDA logic is preserved. Enhancements focus on layout, colors,
# readability, and a calm background image suitable for long reading.
# -----------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    adjusted_rand_score, normalized_mutual_info_score,
    silhouette_score, davies_bouldin_score, mean_absolute_error, mean_squared_error
)
from sklearn.ensemble import GradientBoostingRegressor
# New imports for embeddings
from sentence_transformers import SentenceTransformer

# Add this to the top of your script for Keras compatibility
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# -----------------------------------------------------------------------------
# Page Config (first Streamlit call)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Drug Performance Analytics Suite",
    page_icon="💊",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Light, comfortable theme + background image
# -----------------------------------------------------------------------------
DEFAULT_BG = "https://images.unsplash.com/photo-1551281044-8b59f0d0bde5?q=80&w=1920&auto=format&fit=crop"
try:
    BG_IMAGE_URL = st.secrets["BG_IMAGE_URL"]
except Exception:
    BG_IMAGE_URL = os.environ.get("BG_IMAGE_URL", DEFAULT_BG)

st.markdown(f"""
<style>
/* Global base */
html, body, [data-testid="stAppViewContainer"], .stApp {{
    color: #1a202c; /* Gray-900 */
    background: #f7fafc; /* Gray-50 */
}}

/* Background image with subtle overlay for readability */
.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    background-image: url('{BG_IMAGE_URL}');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    opacity: 0.15; /* keep it subtle */
    z-index: -2;
}}
.stApp::after {{
    content: "";
    position: fixed;
    inset: 0;
    background: linear-gradient(180deg, rgba(255,255,255,0.6), rgba(247,250,252,0.9));
    z-index: -1;
}}

/* Typography */
h1, h2, h3 {{
    font-weight: 800 !important;
    letter-spacing: 0.2px;
    color: #0f172a; /* slate-900 */
}}
p, li, label, .stMarkdown {{
    font-size: 1rem;
    line-height: 1.6;
}}

/* Cards / containers */
.block-container {{
    padding-top: 2rem;
}}
.card {{
    background: #ffffffcc;
    border: 1px solid #edf2f7;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
}}

/* Dataframes */
[data-testid="stDataFrame"] {{
    background: #ffffff;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
}}

/* Metrics */
[data-testid="stMetricValue"] {{
    color: #0f766e !important; /* teal-800 */
    font-weight: 800 !important;
}}

/* Buttons (primary) */
.stButton > button, button[kind="primaryFormSubmit"] {{
    background: linear-gradient(90deg, #10b981, #059669); /* green gradient */
    color: white;
    font-weight: 700;
    border-radius: 12px;
    border: none;
    padding: 0.6rem 1rem;
    transition: transform .05s ease-in-out, box-shadow .15s ease;
    box-shadow: 0 6px 14px rgba(16,185,129,0.25);
}}
.stButton > button:hover,
button[kind="primaryFormSubmit"]:hover {{
    transform: translateY(-1px);
    box-shadow: 0 10px 20px rgba(16,185,129,0.28);
}}

/* Tabs */
.stTabs [data-baseweb="tab"] {{
    font-weight: 700;
}}

/* Divider */
hr {{
    border: none;
    border-top: 1px solid #e2e8f0;
}}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 💊 Drug Performance Analytics Suite")
    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "📈 Exploratory Analysis (EDA)",
            "🤖 Classification: Predict Condition",
            "🔬 Clustering: Group by Condition",
            "🔮 Regression: Performance Prediction"
        ],
        index=0
    )
    st.markdown("---")
    st.caption("Use the pages below to explore the workflows interactively.")

# -----------------------------------------------------------------------------
# Data Loading (cached)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Loads the drug performance dataset."""
    try:
        path = kagglehub.dataset_download('thedevastator/drug-performance-evaluation')
        drug_csv_path = os.path.join(path, 'Drug.csv')
        df_ = pd.read_csv(drug_csv_path)
        return df_
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

# -----------------------------------------------------------------------------
# Helper: CSV download
# -----------------------------------------------------------------------------
def download_df(df_to_download: pd.DataFrame, label: str):
    csv = df_to_download.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"⬇️ Download {label} (CSV)",
        data=csv,
        file_name=f"{label.replace(' ', '_').lower()}.csv",
        mime="text/csv",
        use_container_width=True
    )

# -----------------------------------------------------------------------------
# Page: Overview
# -----------------------------------------------------------------------------
if page == "🏠 Overview":
    st.title("Drug Performance Analytics Suite")
    st.subheader("Interactive analysis and modeling of patient drug feedback.")
    st.write(
        """
        Explore end-to-end workflows:
        - **Exploratory Data Analysis** for understanding the data  
        - **Classification** to predict *Condition* - **Clustering** to map *Condition* groups  
        - **Regression** to predict a composite *performance* score
        """
    )
    st.markdown("---")
    cols = st.columns(4)
    with cols[0]:
        st.metric("Records", f"{df.shape[0] if df is not None else '—'}")
    with cols[1]:
        st.metric("Columns", f"{df.shape[1] if df is not None else '—'}")
    with cols[2]:
        st.metric("Models", "3")
    with cols[3]:
        st.metric("Interactive Pages", "4")

    st.markdown("---")
    st.markdown("### Dataset Preview")
    if df is not None:
        st.dataframe(df.head(), use_container_width=True)
        download_df(df.head(1000) if len(df) > 1000 else df, "dataset_sample")

# -----------------------------------------------------------------------------
# Page: Exploratory Analysis (EDA) — Features Restored
# -----------------------------------------------------------------------------
if page == "📈 Exploratory Analysis (EDA)" and df is not None:
    st.title("Understanding the Data 💊")
    st.markdown("What makes a drug successful from the patient's perspective? Interact with the plots and explore the relationships.")
    st.divider()

    df_eda = df.copy()
    st.header("Sample")
    st.dataframe(df_eda.head(), use_container_width=True)

    with st.expander("Technical summary"):
        st.subheader("Shape")
        st.write(f"**{df_eda.shape[0]} rows**, **{df_eda.shape[1]} columns**.")
        st.subheader("Data Types")
        st.write(df_eda.dtypes)
        st.subheader("Statistics")
        st.dataframe(df_eda.select_dtypes(include="number").describe(), use_container_width=True)
    st.divider()

    st.header("Rating Distributions 📈")
    numeric_df = df_eda.select_dtypes(include="number")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(numeric_df.columns[:3]):
        sns.histplot(numeric_df[col], kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(f'Distribution of {col}', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    st.divider()

    st.header("Correlations 🔗")
    corr = df_eda.select_dtypes(include="number").corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.3, ax=ax)
    st.pyplot(fig)
    st.divider()

    # --- Box and Violin Plots Restored ---
    st.header("Metric Deep-Dive 🗣️")
    cols_to_analyze = ["Effective", "Satisfaction", "EaseOfUse"]
    tab1, tab2, tab3 = st.tabs([f"📊 {col}" for col in cols_to_analyze])
    tabs_dict = {"Effective": tab1, "Satisfaction": tab2, "EaseOfUse": tab3}
    for col, tab in tabs_dict.items():
        with tab:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Box Plot**")
                fig_b, ax_b = plt.subplots(figsize=(6, 5))
                sns.boxplot(y=df_eda[col], ax=ax_b, color='lightblue')
                st.pyplot(fig_b)
            with c2:
                st.markdown("**Violin Plot**")
                fig_v, ax_v = plt.subplots(figsize=(6, 5))
                sns.violinplot(data=df_eda, y=col, inner="quartile", color='lightgreen')
                st.pyplot(fig_v)
    st.divider()

    # --- Pair Plot Restored ---
    st.header("Summary")
    st.markdown("**Effectiveness** is the main driver of **Satisfaction**, with **Ease of Use** as a meaningful secondary factor.")
    with st.expander("360° view (pair plot)"):
        df_sample = df_eda.sample(n=min(500, len(df_eda)), random_state=42)
        g = sns.pairplot(df_sample.select_dtypes(include='number'))
        g.fig.suptitle("Relationships Overview", y=1.02)
        st.pyplot(g.fig)
    st.markdown("---")
    download_df(df_eda, "eda_filtered_data")

# -----------------------------------------------------------------------------
# Page: Classification — Live Prediction Restored
# -----------------------------------------------------------------------------
if page == "🤖 Classification: Predict Condition" and df is not None:
    st.title("Predicting the Condition")
    st.markdown("Logistic Regression with text embeddings for 'Information', one-hot encoding for 'Drug', and scaling for numerics.")

    @st.cache_resource
    def train_classification_model(df_in):
        with st.spinner("Training classification model... 🧠"):
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            X = df_in.drop('Condition', axis=1)
            y = df_in['Condition']
            le_cond = LabelEncoder()
            y_encoded = le_cond.fit_transform(y)
            
            num_cols = ['EaseOfUse', 'Effective', 'Satisfaction']
            scaler = StandardScaler()
            X_num = scaler.fit_transform(X[num_cols].fillna(0))
            
            cat_cols = ['Drug']
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            X_cat = encoder.fit_transform(X[cat_cols].astype(str))
            
            info_embeddings = embedding_model.encode(df_in['Information'].tolist())
            
            X_processed = np.hstack([X_num, X_cat, info_embeddings])
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_encoded, test_size=0.2, random_state=42)
            
            model = LogisticRegression(max_iter=1000, solver='lbfgs')
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            results = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            return model, le_cond, encoder, scaler, embedding_model, results

    model, le_cond, encoder, scaler, embedding_model, results = train_classification_model(df)
    
    st.subheader("Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{results['Accuracy']*100:.2f}%")
    col2.metric("Precision", f"{results['Precision']*100:.2f}%")
    col3.metric("Recall", f"{results['Recall']*100:.2f}%")
    col4.metric("F1 Score", f"{results['F1 Score']*100:.2f}%")
    st.divider()

    # --- Live Prediction Tool Restored to Original ---
    st.subheader("Live Prediction")
    with st.form("prediction_form_clf"):
        st.markdown("#### Enter Patient Feedback")
        drug_input = st.selectbox("Select Drug", options=sorted(df['Drug'].unique()))
        info_input = st.selectbox("Information Provided", options=df['Information'].unique())
        ease_input = st.slider("Ease of Use", 1.0, 5.0, 4.0, 0.1)
        eff_input = st.slider("Effectiveness", 1.0, 5.0, 4.0, 0.1)
        sat_input = st.slider("Satisfaction", 1.0, 5.0, 4.0, 0.1)
        submitted = st.form_submit_button("Predict", use_container_width=True, type="primary")

        if submitted:
            user_cat = pd.DataFrame([[drug_input]], columns=['Drug'])
            user_num = pd.DataFrame([[ease_input, eff_input, sat_input]], columns=['EaseOfUse', 'Effective', 'Satisfaction'])
            
            user_cat_enc = encoder.transform(user_cat)
            user_num_scl = scaler.transform(user_num)
            user_info_emb = embedding_model.encode([info_input])
            
            user_features = np.hstack([user_num_scl, user_cat_enc, user_info_emb])
            
            pred_enc = model.predict(user_features)
            pred_proba = model.predict_proba(user_features).max()
            predicted_condition = le_cond.inverse_transform(pred_enc)[0]
            
            st.success(f"**Predicted Condition:** {predicted_condition}")
            st.info(f"**Confidence:** {pred_proba*100:.2f}%")
    st.markdown("---")
    download_df(df, "classification_input_view")

# -----------------------------------------------------------------------------
# Page: Clustering — Live Prediction Restored
# -----------------------------------------------------------------------------
if page == "🔬 Clustering: Group by Condition" and df is not None:
    st.title('Clustering & Condition Mapping')
    st.markdown("Using K-Means to demonstrate how unsupervised learning can identify distinct groups within the data.")
    st.divider()
    
    df_processed = df.copy()
    X_condition_encoded = pd.get_dummies(df_processed[['Condition']])
    num_unique_conditions = len(X_condition_encoded.columns)
    kmeans = KMeans(n_clusters=num_unique_conditions, random_state=42, n_init='auto')
    df_processed['Cluster'] = kmeans.fit_predict(X_condition_encoded)
    st.dataframe(df_processed.head(10), use_container_width=True)
    st.divider()

    st.header("Clustering Metrics")
    y_true_labels = df_processed['Condition']
    y_pred_clusters = df_processed['Cluster']
    ari = adjusted_rand_score(y_true_labels, y_pred_clusters)
    nmi = normalized_mutual_info_score(y_true_labels, y_pred_clusters)
    c1, c2 = st.columns(2)
    c1.metric(label="Adjusted Rand (→ 1.0)", value=f"{ari:.4f}")
    c2.metric(label="NMI (→ 1.0)", value=f"{nmi:.4f}")
    st.divider()

    # --- Live Prediction Tool Restored ---
    st.header("Live Prediction")
    unique_conditions = sorted(df_processed['Condition'].unique())
    selected_condition = st.selectbox('Select a Condition:', unique_conditions)
    if st.button('Predict Cluster'):
        cluster_num = df_processed[df_processed['Condition'] == selected_condition]['Cluster'].iloc[0]
        st.metric(label=f"Predicted Cluster for {selected_condition}", value=f"Cluster {cluster_num}")
    st.markdown("---")
    download_df(df_processed[['Condition','Cluster']], "clusters_by_condition")

# -----------------------------------------------------------------------------
# Page: Regression — Parameters Hardcoded, Live Prediction Restored
# -----------------------------------------------------------------------------
if page == "🔮 Regression: Performance Prediction" and df is not None:
    st.title("Predict Composite Performance (1–5)")
    st.write("A Gradient Boosting model predicts a **performance** score from **Effective** and **EaseOfUse**.")
    st.divider()

    df_initial = df.copy()
    columns_to_drop = ['Indication', 'Type', 'Information', 'Reviews']
    df_processed = df_initial.drop(columns=columns_to_drop)
    df_encoded = df_processed.copy()
    
    categorical_cols = ['Drug', 'Condition']
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le
    
    # --- Performance Weight and Train/Test Split Hardcoded ---
    weight_effective = 0.7
    test_size_fraction = 0.2
    st.info(f"Performance score is calculated with a **{weight_effective*100:.0f}%** weight on 'Effective'.")
    st.info(f"Data is split into an **{int((1-test_size_fraction)*100)}%** training set and **{int(test_size_fraction*100)}%** testing set.")
    
    df_encoded['performance'] = (weight_effective * df_encoded['Effective']) + ((1.0 - weight_effective) * df_encoded['EaseOfUse'])
    X = df_encoded[['Drug', 'Condition', 'Effective', 'EaseOfUse']].copy()
    y = df_encoded['performance']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size_fraction, random_state=42)

    with st.spinner("Training the model... 🌳"):
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(x_train, y_train)
    
    st.subheader("Model Performance")
    y_pred_gb = gb_model.predict(x_test)
    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    mse_gb = mean_squared_error(y_test, y_pred_gb)
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("MAE", f"{mae_gb:.4f}")
    m_col2.metric("MSE", f"{mse_gb:.4f}")
    st.divider()

    # --- Live Prediction Tool Restored ---
    st.header("Live Prediction")
    with st.form("prediction_form_reg"):
        input_col1, input_col2 = st.columns(2)
        with input_col1:
            drug = st.selectbox("Drug Name", options=encoders['Drug'].classes_)
            condition = st.selectbox("Condition", options=encoders['Condition'].classes_)
        with input_col2:
            effective = st.slider("Effectiveness (1-5)", 1.0, 5.0, 4.0, 0.1)
            ease_of_use = st.slider("Ease of Use (1-5)", 1.0, 5.0, 4.0, 0.1)
        
        submitted_reg = st.form_submit_button("Predict Performance")

        if submitted_reg:
            drug_encoded = encoders['Drug'].transform([drug])[0]
            condition_encoded = encoders['Condition'].transform([condition])[0]
            feature_list = [drug_encoded, condition_encoded, effective, ease_of_use]
            features = np.array(feature_list).reshape(1, -1)
            prediction = gb_model.predict(features)[0]
            st.success(f"**Predicted Performance Score (1-5):** {prediction:.2f}")
    st.markdown("---")
    download_df(df, "regression_input_view")

# -----------------------------------------------------------------------------
# Fallback
# -----------------------------------------------------------------------------
if df is None and page != "🏠 Overview":
    st.warning("Dataset could not be loaded. Please check your Kaggle credentials/connectivity and try again.")