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
# New imports for embeddings and similarity calculation
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


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
# (soft whites, clear contrast, large readable type)
# -----------------------------------------------------------------------------
# SAFE secrets access: fall back to a default image if secrets.toml is missing
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
# Sidebar Navigation (concise, no distracting language)
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
    """Loads the drug performance dataset (original structure)."""
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
        
        The layout, color palette, and typography are tuned for comfortable reading with a calm background image that stays subtle.
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
# Page: Exploratory Analysis (EDA) — core logic preserved
# -----------------------------------------------------------------------------
if page == "📈 Exploratory Analysis (EDA)" and df is not None:

    st.title("Understanding the Data 💊")
    st.markdown("""
    What makes a drug successful from the patient's perspective?  
    Interact with the plots and explore the relationships.
    """)
    st.divider()

    # Optional filter (UI-only; analysis logic unchanged)
    with st.expander("Optional: Filter the data for EDA"):
        colf1, colf2 = st.columns(2)
        with colf1:
            selected_drugs = st.multiselect("Filter by Drug", sorted(df['Drug'].unique()))
        with colf2:
            selected_conditions = st.multiselect("Filter by Condition", sorted(df['Condition'].unique()))
        df_eda = df.copy()
        if selected_drugs:
            df_eda = df_eda[df_eda['Drug'].isin(selected_drugs)]
        if selected_conditions:
            df_eda = df_eda[df_eda['Condition'].isin(selected_conditions)]
        st.caption(f"Rows after filter: {len(df_eda)}")

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

    # Distribution (preserved)
    st.header("Rating Distributions 📈")
    numeric_df = df_eda.select_dtypes(include="number")
    if len(numeric_df.columns) >= 3:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, col in enumerate(numeric_df.columns[:3]):
            sns.histplot(numeric_df[col], kde=True, ax=axes[i], color='skyblue')
            axes[i].set_title(f'Distribution of {col}', fontsize=14)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns to render histograms.")
    st.divider()

    # Correlation (preserved)
    st.header("Correlations 🔗")
    corr = df_eda.select_dtypes(include="number").corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(4, 2))
    sns.heatmap(
        corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.3, ax=ax,
        annot_kws={"size": 8}
    )
    ax.set_title('Correlation Heatmap', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=6)
    st.pyplot(fig)
    st.divider()

    # Deep dive tabs (preserved)
    st.header("Metric Deep-Dive 🗣️")
    cols_to_analyze = ["Effective", "Satisfaction", "EaseOfUse"]
    tab1, tab2, tab3 = st.tabs([f"📊 {col}" for col in cols_to_analyze])
    tabs_dict = {"Effective": tab1, "Satisfaction": tab2, "EaseOfUse": tab3}
    for col, tab in tabs_dict.items():
        with tab:
            st.subheader(f"'{col}'")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Box Plot**")
                fig_b, ax_b = plt.subplots(figsize=(6, 5))
                sns.boxplot(y=df_eda[col], ax=ax_b, color='lightblue')
                ax_b.set_title(f"Box Plot: {col}")
                st.pyplot(fig_b)
            with c2:
                st.markdown("**Violin Plot**")
                fig_v, ax_v = plt.subplots(figsize=(6, 5))
                sns.violinplot(data=df_eda, x=col, inner="quartile", color='lightgreen')
                ax_v.set_title(f"Violin Plot: {col}")
                st.pyplot(fig_v)
    st.divider()

    # Relationships (preserved)
    st.header("What Drives Satisfaction? 💥")
    tab_eff, tab_ease = st.tabs(["Satisfaction vs. Effective", "Satisfaction vs. EaseOfUse"])
    with tab_eff:
        fig, ax = plt.subplots(figsize=(5,2))
        sns.regplot(data=df_eda, x="Effective", y="Satisfaction", ax=ax, line_kws={"color": "red"}, scatter_kws={'alpha':0.4})
        ax.set_title("Effectiveness vs Satisfaction", fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=6)
        st.pyplot(fig)
    with tab_ease:
        fig, ax = plt.subplots(figsize=(5,2))
        sns.regplot(data=df_eda, x="EaseOfUse", y="Satisfaction", ax=ax, line_kws={"color": "red"}, scatter_kws={'alpha':0.5})
        ax.set_title("Ease of Use vs Satisfaction", fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=6)
        st.pyplot(fig)
    st.divider()

    st.header("Summary")
    st.markdown("""
    **Effectiveness** is the main driver of **Satisfaction**, with **Ease of Use** as a meaningful secondary factor.
    """)
    with st.expander("360° view (pair plot)"):
        df_sample = df_eda.sample(n=min(500, len(df_eda)), random_state=42)
        g = sns.pairplot(df_sample, hue="Satisfaction", diag_kind="kde", palette="viridis")
        g.fig.suptitle("Relationships Overview", y=1.02)
        st.pyplot(g.fig)
    st.markdown("---")
    download_df(df_eda, "eda_filtered_data")

# -----------------------------------------------------------------------------
# Page: Classification — Predict Condition (LOGIC UPDATED WITH EMBEDDINGS)
# -----------------------------------------------------------------------------
if page == "🤖 Classification: Predict Condition" and df is not None:
    st.title("Predicting the Condition with Text Embeddings")
    st.markdown("""
    This model uses **text embeddings** for the 'Information' column to capture its semantic meaning. 
    It combines these embeddings with scaled numerical features and one-hot encoded drug data to train a Logistic Regression classifier.
    """)

    # --- New Functions for Embedding and Model Training ---
    @st.cache_resource
    def load_embedding_model():
        """Loads the SentenceTransformer model and caches it."""
        return SentenceTransformer('all-MiniLM-L6-v2')

    @st.cache_data
    def generate_embeddings(_model, sentences):
        """Generates and caches embeddings for a list of sentences."""
        return _model.encode(sentences)

    @st.cache_resource
    def train_classification_model(df_in, _embedding_model):
        """Preprocesses data, generates embeddings, trains, and evaluates the model."""
        with st.spinner("Training classification model... This may take a moment. 🧠"):
            X = df_in.drop('Condition', axis=1)
            y = df_in['Condition']
            
            # 1. Target Encoding
            le_cond = LabelEncoder()
            y_encoded = le_cond.fit_transform(y)
            
            # 2. Feature Processing
            # Numerical features
            num_cols = ['EaseOfUse', 'Effective', 'Satisfaction']
            scaler = StandardScaler()
            X_num = scaler.fit_transform(X[num_cols].fillna(0))
            
            # Categorical feature ('Drug')
            cat_cols = ['Drug']
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            X_cat = encoder.fit_transform(X[cat_cols].astype(str))
            
            # Text feature ('Information') using embeddings
            info_embeddings = generate_embeddings(_embedding_model, df_in['Information'].tolist())
            
            # 3. Combine processed features
            X_processed = np.hstack([X_num, X_cat, info_embeddings])
            
            # 4. Train/Test Split and Model Training
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_encoded, test_size=0.2, random_state=42
            )
            model = LogisticRegression(max_iter=1000, solver='lbfgs')
            model.fit(X_train, y_train)
            
            # 5. Evaluation
            y_pred = model.predict(X_test)
            results = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
                "Embeddings": info_embeddings # Return all embeddings for similarity search
            }
            return model, le_cond, encoder, scaler, results

    # --- Main Page Logic ---
    embedding_model = load_embedding_model()
    model, le_cond, encoder, scaler, results = train_classification_model(df, embedding_model)

    st.subheader("Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{results['Accuracy']*100:.2f}%")
    col2.metric("Precision", f"{results['Precision']*100:.2f}%")
    col3.metric("Recall", f"{results['Recall']*100:.2f}%")
    col4.metric("F1 Score", f"{results['F1 Score']*100:.2f}%")
    st.divider()

    # --- New Section: Information Similarity Explorer ---
    st.subheader("Explore 'Information' Similarity")
    st.info("See how embeddings group similar phrases together. Select a phrase to find its closest matches.")
    
    all_info_phrases = df['Information'].unique().tolist()
    selected_phrase = st.selectbox("Select an information phrase to analyze:", options=all_info_phrases)

    if selected_phrase:
        # Get the embedding for the selected phrase
        phrase_embedding = embedding_model.encode([selected_phrase])
        
        # Calculate cosine similarity against all other embeddings
        all_embeddings = results["Embeddings"]
        similarities = cosine_similarity(phrase_embedding, all_embeddings)[0]
        
        # Create a dataframe for display
        sim_df = pd.DataFrame({
            'Information': df['Information'],
            'Similarity': similarities
        })
        
        # Remove the original phrase itself and show top 5 matches
        top_similar = sim_df[sim_df['Information'] != selected_phrase].sort_values(by='Similarity', ascending=False).head(5)
        
        st.write(f"**Top 5 most similar phrases to:** *'{selected_phrase}'*")
        st.dataframe(top_similar, use_container_width=True)
    st.divider()

    # --- Updated Section: Live Prediction ---
    st.subheader("Live Prediction")
    with st.container():
        with st.form("prediction_form"):
            st.markdown("#### Enter Patient Feedback")
            c1, c2 = st.columns(2)
            with c1:
                drug_input = st.selectbox("Select Drug", options=sorted(df['Drug'].unique()))
                info_input = st.text_input("Information Provided", value="No information provided")
            with c2:
                ease_input = st.slider("Ease of Use", 1, 5, 4)
                eff_input = st.slider("Effectiveness", 1, 5, 4)
                sat_input = st.slider("Satisfaction", 1, 5, 4)
            
            submitted = st.form_submit_button("Predict", use_container_width=True, type="primary")

            if submitted:
                # Process inputs
                user_cat = pd.DataFrame([[drug_input]], columns=['Drug'])
                user_num = pd.DataFrame([[ease_input, eff_input, sat_input]], columns=['EaseOfUse', 'Effective', 'Satisfaction'])
                
                # Transform features
                user_cat_enc = encoder.transform(user_cat)
                user_num_scl = scaler.transform(user_num)
                user_info_emb = embedding_model.encode([info_input])
                
                # Combine features in the correct order
                user_features = np.hstack([user_num_scl, user_cat_enc, user_info_emb])
                
                # Make prediction
                pred_enc = model.predict(user_features)
                pred_proba = model.predict_proba(user_features).max()
                predicted_condition = le_cond.inverse_transform(pred_enc)[0]
                
                st.success(f"**Predicted Condition:** {predicted_condition}")
                st.info(f"**Confidence:** {pred_proba*100:.2f}%")

    st.markdown("---")
    download_df(df[['Drug','Information','EaseOfUse','Effective','Satisfaction','Condition']], "classification_input_view")


# -----------------------------------------------------------------------------
# Page: Clustering — KMeans on Condition (core logic preserved)
# -----------------------------------------------------------------------------
if page == "🔬 Clustering: Group by Condition" and df is not None:
    st.title('Clustering & Condition Mapping')

    st.header("DataFrame")
    st.dataframe(df.head(5), use_container_width=True)

    st.header("Preprocessed DataFrame")
    st.write("""
    Focus on **'Condition'** for grouping. Remove **'Indication'**, **'Type'**, and **'Reviews'**.
    """)
    df_processed = df.drop(columns=['Indication', 'Type', 'Reviews']).copy()
    st.dataframe(df_processed.head(5), use_container_width=True)

    st.header("K-Means Clustering")
    st.success("""
    One-hot encode **'Condition'** and fit K-Means to assign a label for each unique condition.
    """)

    X_condition_encoded = pd.get_dummies(df_processed[['Condition']])
    num_unique_conditions = len(X_condition_encoded.columns)
    kmeans = KMeans(n_clusters=num_unique_conditions, random_state=42, n_init='auto')
    df_processed['Cluster'] = kmeans.fit_predict(X_condition_encoded)
    st.dataframe(df_processed.head(10), use_container_width=True)

    st.header("Live Prediction")
    unique_conditions = sorted(df_processed['Condition'].unique())
    selected_condition = st.selectbox('Select a Condition:', unique_conditions)
    if st.button('Predict Cluster'):
        cluster_num = df_processed[df_processed['Condition'] == selected_condition]['Cluster'].iloc[0]
        st.metric(label=f"Predicted Cluster for {selected_condition}", value=f"Cluster {cluster_num}")
        st.info(f"Assigned label **'{cluster_num}'** for **'{selected_condition}'**.")

    st.header("Clustering Metrics")
    y_true_labels = df_processed['Condition']
    y_pred_clusters = df_processed['Cluster']
    ari = adjusted_rand_score(y_true_labels, y_pred_clusters)
    nmi = normalized_mutual_info_score(y_true_labels, y_pred_clusters)
    sil_score = silhouette_score(X_condition_encoded, y_pred_clusters)
    db_score = davies_bouldin_score(X_condition_encoded, y_pred_clusters)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(label="Adjusted Rand (→ 1.0)", value=f"{ari:.4f}")
    c2.metric(label="NMI (→ 1.0)", value=f"{nmi:.4f}")
    c3.metric(label="Silhouette (→ 1.0)", value=f"{sil_score:.4f}")
    c4.metric(label="Davies-Bouldin (→ 0.0)", value=f"{db_score:.4f}")

    st.markdown("---")
    st.header("Outcome")
    st.success("""
    A clear one-to-one mapping between conditions and cluster labels is achieved for downstream tasks.
    """)
    st.markdown("---")
    download_df(df_processed[['Condition','Cluster']], "clusters_by_condition")

# -----------------------------------------------------------------------------
# Page: Regression — Gradient Boosting (core logic preserved; UI enhanced)
# -----------------------------------------------------------------------------
if page == "🔮 Regression: Performance Prediction" and df is not None:

    st.markdown("""
    <style>
    .prediction-container {
        border: 2px solid #059669;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: rgba(5,150,105,0.06);
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Predict Composite Performance (1–5)")
    st.write("Create a **performance** target from **Effective** and **EaseOfUse**, then train a Gradient Boosting model.")

    @st.cache_data
    def load_data_from_kaggle():
        """Downloads and caches the dataset from Kaggle Hub."""
        try:
            dataset_path = kagglehub.dataset_download('thedevastator/drug-performance-evaluation')
            drug_csv_path = os.path.join(dataset_path, 'Drug.csv')
            df_local = pd.read_csv(drug_csv_path)
            return df_local
        except Exception as e:
            st.error(f"Error loading data from Kaggle Hub: {e}")
            return None

    df_initial = load_data_from_kaggle()

    if df_initial is not None:
        st.subheader("Initial Data")
        st.dataframe(df_initial.head(), use_container_width=True)

        st.subheader("After Preprocessing")
        st.write("Remove **'Indication'**, **'Type'**, **'Information'**, and **'Reviews'**.")
        columns_to_drop = ['Indication', 'Type', 'Information', 'Reviews']
        df_processed = df_initial.drop(columns=columns_to_drop)
        st.dataframe(df_processed.head(), use_container_width=True)

        st.subheader("After Encoding Text Features")
        st.write("Convert 'Drug' and 'Condition' to numeric labels.")
        df_encoded = df_processed.copy()

        categorical_cols = ['Drug', 'Condition']
        encoders = {}
        for col in categorical_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                encoders[col] = le

        st.session_state['encoders'] = encoders
        st.dataframe(df_encoded.head(), use_container_width=True)

        st.subheader("Create 'performance' Target")
        st.write("Weight **Effective** vs **EaseOfUse**.")
        weight_effective = st.slider(
            "Importance Weight for 'Effective' Score:", 0.0, 1.0, 0.7, 0.05
        )
        df_encoded['performance'] = (weight_effective * df_encoded['Effective']) + ((1.0 - weight_effective) * df_encoded['EaseOfUse'])
        st.dataframe(df_encoded.head(), use_container_width=True)

        # Training uses exactly the 4 features used in live prediction (preserved)
        X = df_encoded[['Drug', 'Condition', 'Effective', 'EaseOfUse']].copy()
        y = df_encoded['performance']

        st.subheader("Train / Test Split")
        test_size = st.slider("Test Set Size (%):", 10, 50, 20, 5)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=(test_size/100.0), random_state=42)

        col1, col2 = st.columns(2)
        col1.metric("Training Set Shape", str(x_train.shape))
        col2.metric("Testing Set Shape", str(x_test.shape))

        if st.button("Train Gradient Boosting Model"):
            with st.spinner("Training the model... 🌳"):
                gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                gb_model.fit(x_train, y_train)
                st.session_state['model'] = gb_model
            st.success("Model trained successfully!")

        if 'model' in st.session_state:
            st.subheader("Model Performance")
            y_pred_gb = st.session_state.model.predict(x_test)
            mae_gb = mean_absolute_error(y_test, y_pred_gb)
            mse_gb = mean_squared_error(y_test, y_pred_gb)

            m_col1, m_col2 = st.columns(2)
            m_col1.metric("MAE", f"{mae_gb:.4f}")
            m_col2.metric("MSE", f"{mse_gb:.4f}")

            with st.expander("Feature importances"):
                fig_imp, ax_imp = plt.subplots(figsize=(5,3))
                importances = st.session_state.model.feature_importances_
                feat_names = X.columns
                ax_imp.bar(feat_names, importances)
                ax_imp.set_title("Feature Importances")
                ax_imp.set_ylabel("Importance")
                plt.xticks(rotation=15)
                st.pyplot(fig_imp)

            st.markdown("---")
            st.header("Live Prediction")

            live_encoders = st.session_state.get('encoders', {})

            input_col1, input_col2 = st.columns(2)
            with input_col1:
                drug = st.selectbox("Drug Name", options=live_encoders.get('Drug', LabelEncoder()).classes_)
                condition = st.selectbox("Condition", options=live_encoders.get('Condition', LabelEncoder()).classes_)
            with input_col2:
                effective = st.slider("Effectiveness (1-5)", 1.0, 5.0, 4.0, 0.1)
                ease_of_use = st.slider("Ease of Use (1-5)", 1.0, 5.0, 4.0, 0.1)

            if st.button("Predict Performance"):
                drug_encoded = live_encoders['Drug'].transform([drug])[0]
                condition_encoded = live_encoders['Condition'].transform([condition])[0]
                feature_list = [drug_encoded, condition_encoded, effective, ease_of_use]
                features = np.array(feature_list).reshape(1, -1)
                prediction = st.session_state.model.predict(features)[0]

                with st.container():
                    st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
                    st.metric(label="Predicted Performance (1–5)", value=f"{prediction:.2f}")
                    st.caption(f"Weight Effective: {weight_effective:.2f} | Weight EaseOfUse: {(1-weight_effective):.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Please train a model to enable evaluation and prediction.")

    st.markdown("---")
    if df is not None:
        download_df(df[['Drug','Condition','EaseOfUse','Effective','Satisfaction']], "regression_input_view")

# -----------------------------------------------------------------------------
# Fallback
# -----------------------------------------------------------------------------
if df is None and page != "🏠 Overview":
    st.warning("Dataset could not be loaded. Please check your Kaggle credentials/connectivity and try again.") 