import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import kagglehub

# --- Page Configuration ---
# Set the layout and title for our Streamlit app. This should be the first Streamlit command.
st.set_page_config(
    page_title="Drug Performance Analysis",
    page_icon="💊",
    layout="wide"
)

# --- Data Loading ---
# A function to load our data, with caching to improve performance.
@st.cache_data
def load_data():
    """Loads the drug performance dataset."""
    try:
        # Using Kaggle Hub to download the dataset
        thedevastator_drug_performance_evaluation_path = kagglehub.dataset_download('thedevastator/drug-performance-evaluation')
        drug_csv_path = os.path.join(thedevastator_drug_performance_evaluation_path, 'Drug.csv')
        df = pd.read_csv(drug_csv_path)
        print("Data source import complete.")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

# Only proceed if the dataframe is loaded successfully
if df is not None:

    # --- Introduction ---
    st.title("Unlocking the Secrets of Drug Performance 💊")
    st.markdown("""
    Welcome to our analysis! We're on a mission to understand what *truly* makes a drug successful from a patient's perspective. 
    Is it raw **effectiveness**? The **ease of use**? Or overall **satisfaction**? 
    
    Join us as we dive into the data to uncover the relationships between these key factors and find out what matters most to patients.
    """)
    st.divider()

    # --- Meet the Data ---
    st.header("Meet the Data: Our Case File 📂")
    st.markdown("Before we start our investigation, let's get acquainted with our data. Here's a small sample of the patient feedback we'll be analyzing.")
    st.dataframe(df.head())

    with st.expander("Click here for a technical summary of the dataset"):
        st.subheader("Shape of the Dataset")
        st.write(f"Our dataset contains **{df.shape[0]} rows** (patient reviews) and **{df.shape[1]} columns** (feedback categories).")
        
        st.subheader("Data Types")
        st.write("Here are the data types for each column. We'll be focusing on the numerical ratings.")
        st.write(df.dtypes)
        
        st.subheader("Statistical Overview")
        st.markdown("Let's get a quick statistical summary of the numerical ratings. This gives us a sense of the average scores and their spread.")
        st.dataframe(df.select_dtypes(include="number").describe(), use_container_width=True)
    st.divider()


    # --- Distribution of Ratings ---
    st.header("A First Look: How are the Ratings Distributed? 📈")
    st.markdown("""
    Are patients generally happy or dissatisfied? Are the ratings all over the place or clustered together? 
    Histograms help us see the 'shape' of the data for our main metrics.
    """)
    
    # Create histograms for numerical data
    numeric_df = df.select_dtypes(include="number")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(numeric_df.columns):
        sns.histplot(numeric_df[col], kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(f'Distribution of {col} Ratings', fontsize=14)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    st.pyplot(fig)

    st.info("""
    **Takeaway:** The ratings for **Effective**, **Satisfaction**, and **EaseOfUse** are all skewed to the left, which is great news! 
    This means most patients are giving high scores (typically 4s and 5s) across the board. Satisfaction appears to have the most high ratings.
    """, icon="💡")
    st.divider()


    # --- Correlation Analysis ---
    st.header("Connecting the Dots: How Do Our Metrics Relate? 🔗")
    st.markdown("""
    Now, let's investigate the relationships between our key metrics. A correlation heatmap is the perfect tool for this. 
    A score close to **+1 (bright red)** means two metrics increase together. A score close to **-1 (bright blue)** means one goes up as the other goes down. A score near **0** means little to no relationship.
    """)

    corr = df.select_dtypes(include="number").corr(numeric_only=True)

    # Adjust figsize to make the plot smaller and add annot_kws for smaller font size
    fig, ax = plt.subplots(figsize=(4, 2))
    sns.heatmap(
        corr, 
        annot=True, 
        cmap="coolwarm", 
        fmt=".2f", 
        linewidths=.3, 
        ax=ax,
        annot_kws={"size": 8} # This line makes the annotation font smaller
    )
    plt.title('Correlation Heatmap of Drug Ratings', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=5)
    st.pyplot(fig)

    st.info("""
    **Key Insight:** We see a very strong positive correlation (**0.83**) between **Effective** and **Satisfaction**. 
    This is our first major clue! How well a drug works seems to be tightly linked to how happy a patient is. The link between **EaseOfUse** and **Satisfaction** is also positive (0.68), but not as strong.
    """, icon="🎯")
    st.divider()


    # --- Deep Dive into Each Metric ---
    st.header("A Deeper Look: Patient Voices on Each Metric 🗣️")
    st.markdown("Let's put each rating under the microscope to understand its distribution and spread in more detail.")

    cols_to_analyze = ["Effective", "Satisfaction", "EaseOfUse"]
    # Using st.tabs for a cleaner, more interactive layout
    tab1, tab2, tab3 = st.tabs([f"📊 {col}" for col in cols_to_analyze])

    tabs_dict = {
        "Effective": tab1,
        "Satisfaction": tab2,
        "EaseOfUse": tab3
    }
    
    for col, tab in tabs_dict.items():
        with tab:
            st.subheader(f"Detailed Analysis of '{col}'")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Box Plot
                st.markdown("**Box Plot**")
                st.write("This shows the median (the line in the box), the middle 50% of ratings (the box), and identifies any unusual outlier scores.")
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.boxplot(y=df[col], ax=ax, color='lightblue')
                ax.set_title(f"Box Plot for {col}")
                st.pyplot(fig)

            with col2:
                # Violin Plot
                st.markdown("**Violin Plot**")
                st.write("This combines a box plot with a density plot, showing where the ratings are most concentrated.")
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.violinplot(data=df, x=col, inner="quartile", color='lightgreen')
                ax.set_title(f"Violin Plot for {col}")
                st.pyplot(fig)

    st.info("""
    **Takeaway:** The Violin plots confirm what we saw earlier: the ratings are heavily concentrated at the higher end (4 and 5). 
    The Box plots show that the median score for all three metrics is high, but 'Satisfaction' has the highest median of all.
    """, icon="🧐")
    st.divider()

    # --- The Core Relationships ---
    st.header("The Ultimate Showdown: What Drives Satisfaction? 💥")
    st.markdown("Our heatmap gave us a hint, but let's visualize the relationships directly. A scatter plot helps us see the trend between two variables.")

    tab_eff, tab_ease = st.tabs(["Satisfaction vs. Effective", "Satisfaction vs. EaseOfUse"])

    with tab_eff:
        fig, ax = plt.subplots(figsize=(5,2))
        sns.regplot(data=df, x="Effective", y="Satisfaction", ax=ax, line_kws={"color": "red"}, scatter_kws={'alpha':0.4})
        ax.set_title("The Powerful Link: Effectiveness Drives Satisfaction", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=4)
        st.pyplot(fig)
        st.write("We can clearly see an upward trend. As the **Effectiveness** rating increases, the **Satisfaction** rating almost always increases with it.")

    with tab_ease:
        fig, ax = plt.subplots(figsize=(5, 2))
        sns.regplot(data=df, x="EaseOfUse", y="Satisfaction", ax=ax, line_kws={"color": "red"}, scatter_kws={'alpha':0.5})
        ax.set_title("The Supporting Role: Ease of Use and Satisfaction", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=4)
        st.pyplot(fig)
        st.write("There is still a positive trend here, but it's more spread out than the effectiveness plot. This confirms that while making a drug easy to use is important for satisfaction, it's not as critical as making it effective.")
    st.divider()

    # --- Conclusion ---
    st.header("The Final Verdict: What TRULY Matters? 🏆")
    st.markdown("""
    After analyzing the data from multiple angles, the story is clear.
    
    While patients appreciate a drug that is easy to use, their **overall satisfaction is overwhelmingly driven by the drug's effectiveness.** The data shows a powerful, direct relationship between a drug working well and a patient being happy with it.
    
    ### Final Recommendation:
    For pharmaceutical companies aiming for high patient satisfaction, the primary focus must remain on **maximizing the efficacy of their products**. While usability is a valuable secondary goal, it cannot compensate for a lack of effectiveness.
    """)
    
    with st.expander("View a final Pair Plot for a 360° view of all relationships"):
        st.write("This plot shows every scatter plot and distribution in one grid, confirming our findings.")
        # Using a smaller sample for the pairplot to speed up rendering in Streamlit
        df_sample = df.sample(n=min(500, len(df)))
        g = sns.pairplot(df_sample, hue="Satisfaction", diag_kind="kde", palette="viridis")
        g.fig.suptitle("360° View of All Metric Relationships", y=1.02)
        st.pyplot(g.fig)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- This code assumes 'df' is a pandas DataFrame that has already been loaded. ---
# Example:
# @st.cache_data
# def load_data():
#     # ... your data loading logic here ...
#     return df
# df = load_data()


# --- Model Training Function (Cached for performance) ---
@st.cache_resource
def train_logistic_regression(df):
    """
    Trains a Logistic Regression model based on the provided dataframe and returns
    the trained model, data transformers, and performance metrics.
    """
    # 1. Define Features (X) and Target (y)
    X = df.drop('Condition', axis=1)
    y = df['Condition']

    # 2. Encode the Target Variable
    le_cond = LabelEncoder()
    y_encoded = le_cond.fit_transform(y)

    # 3. Preprocess the Features
    cat_cols = ['Drug', 'Information']
    num_cols = ['EaseOfUse', 'Effective', 'Satisfaction']
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    scaler = StandardScaler()

    X_cat = encoder.fit_transform(X[cat_cols].astype(str))
    X_num = scaler.fit_transform(X[num_cols].fillna(0))
    X_processed = np.hstack([X_num, X_cat])

    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=0.2, random_state=42
    )

    # 5. Train Model
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_train, y_train)
    
    # 6. Evaluate Model
    y_pred = model.predict(X_test)
    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    
    return model, le_cond, encoder, scaler, results


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- This code assumes 'df' is a pandas DataFrame that has already been loaded. ---
# For demonstration, we create a sample 'df' if one isn't found.
try:
    df.head()
except (NameError, AttributeError):
    st.warning("Using a sample DataFrame for this demo. Please ensure your data is loaded.")
    data = {
        'Drug': ['DrugA', 'DrugB', 'DrugC'], 'Information': ['Info1', 'Info2', 'Info3'],
        'EaseOfUse': [4, 5, 3], 'Effective': [5, 4, 3], 'Satisfaction': [5, 4, 3],
        'Condition': ['ConditionX', 'ConditionY', 'ConditionZ']
    }
    df = pd.DataFrame(data)

# --- Model Training Function (Cached for performance) ---
@st.cache_resource
def train_logistic_regression(df):
    X = df.drop('Condition', axis=1)
    y = df['Condition']
    le_cond = LabelEncoder()
    y_encoded = le_cond.fit_transform(y)
    cat_cols = ['Drug', 'Information']
    num_cols = ['EaseOfUse', 'Effective', 'Satisfaction']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    scaler = StandardScaler()
    X_cat = encoder.fit_transform(X[cat_cols].astype(str))
    X_num = scaler.fit_transform(X[num_cols].fillna(0))
    X_processed = np.hstack([X_num, X_cat])
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    return model, le_cond, encoder, scaler, results

# --- Main Streamlit App Logic for Classification ---

# Only run if the dataframe exists
if 'df' in locals() and df is not None:

    # --- Custom CSS for a BRIGHT GREEN Button ---
    st.markdown("""
    <style>
        /* This targets the form submit button specifically */
        div[data-testid="stForm"] button[kind="primaryFormSubmit"] {
            background-color: #00C853; /* A bright, modern green */
            color: white;
            border: none;
            border-radius: 5px;
            box-shadow: 0 4px 14px 0 rgba(0, 118, 255, 0.39);
        }
        div[data-testid="stForm"] button[kind="primaryFormSubmit"]:hover {
            background-color: #00E676; /* A slightly lighter green for hover */
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🤖 Machine Learning: Predicting the Condition")

    st.markdown("""
    Our goal is to accurately predict a patient's medical condition based on their feedback. We use a Logistic Regression model trained on their ratings and the specific drug prescribed.
    """)

    model, le_cond, encoder, scaler, results = train_logistic_regression(df)

    st.subheader("Model Performance Report Card 📊")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{results['Accuracy']*100:.2f}%")
    col2.metric("Precision", f"{results['Precision']*100:.2f}%")
    col3.metric("Recall", f"{results['Recall']*100:.2f}%")
    col4.metric("F1 Score", f"{results['F1 Score']*100:.2f}%")

    st.divider()

    # --- Live Prediction Tool ---
    st.subheader("🧪 Live Prediction Tool")
    
    # The container now holds only the form, which will take the full width.
    with st.container(border=True):
        with st.form("prediction_form"):
            st.markdown("#### Enter Patient Feedback")
            
            # The columns are removed, so inputs are arranged vertically.
            drug_input = st.selectbox("Select Drug", options=sorted(df['Drug'].unique()))
            info_input = st.selectbox("Information Provided", options=df['Information'].unique())
            ease_input = st.slider("Ease of Use Rating", 1, 5, 4)
            eff_input = st.slider("Effectiveness Rating", 1, 5, 4)
            sat_input = st.slider("Satisfaction Rating", 1, 5, 4)
            
            # The button is styled by the CSS above.
            submitted = st.form_submit_button(
                "✨ Predict Condition Now!", 
                use_container_width=True, 
                type="primary"
            )

            if submitted:
                user_cat_data = pd.DataFrame([[drug_input, info_input]], columns=['Drug', 'Information'])
                user_num_data = pd.DataFrame([[ease_input, eff_input, sat_input]], columns=['EaseOfUse', 'Effective', 'Satisfaction'])
                user_cat_encoded = encoder.transform(user_cat_data)
                user_num_scaled = scaler.transform(user_num_data)
                user_features = np.hstack([user_num_scaled, user_cat_encoded])
                
                prediction_encoded = model.predict(user_features)
                prediction_proba = model.predict_proba(user_features).max()
                predicted_condition = le_cond.inverse_transform(prediction_encoded)[0]

                st.success(f"**Predicted Condition:** {predicted_condition}")
                st.info(f"**Model Confidence:** {prediction_proba*100:.2f}%")

else:
    st.warning("Please load the dataset first to use the classification tool.") 


import streamlit as st
import pandas as pd
import os
import kagglehub
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, davies_bouldin_score
import sklearn

# --- Page Configuration ---
st.set_page_config(
    page_title="Drug Performance Clustering",
    page_icon="🔬",
    layout="wide"
)

# --- Custom Styling for the Button ---
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #007BFF;
    color: white;
    font-size: 16px;
    font-weight: bold;
    padding: 10px 24px;
    border-radius: 8px;
    border: none;
    transition-duration: 0.4s;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
}
div.stButton > button:first-child:hover {
    background-color: #0056b3;
    color: white;
    box-shadow: 0 6px 12px 0 rgba(0,0,0,0.3);
}
</style>""", unsafe_allow_html=True)


# --- Main Title ---
st.title('🔬 Drug Performance Clustering & Prediction')

# --- Data Loading Function ---
@st.cache_data
def load_data_from_kaggle():
    try:
        dataset_path = kagglehub.dataset_download('thedevastator/drug-performance-evaluation')
        drug_csv_path = os.path.join(dataset_path, 'Drug.csv')
        df = pd.read_csv(drug_csv_path)
        return df
    except Exception as e:
        st.error(f"Error loading data from Kaggle Hub: {e}")
        return None

# --- Main App Logic ---
df_initial = load_data_from_kaggle()

if df_initial is not None:
    # --- Initial DataFrame Display ---
    st.header("DataFrame")
    st.dataframe(df_initial.head(5))
    
    # --- Data Preprocessing ---
    st.header("Preprocessed DataFrame")
    
    """
        The goal is to cluster drugs based purely on their prescribed **'Condition'**. Columns like **'Indication'**, 
        **'Type'**, and **'Reviews'** are removed because they represent secondary information or user opinions, 
        which could interfere with the primary objective of grouping by the core medical purpose.
        """
    
    
    df_processed = df_initial.drop(columns=['Indication', 'Type', 'Reviews']).copy()
    st.write("The DataFrame for clustering task:")
    st.dataframe(df_processed.head(5))

    # --- K-Means Clustering Implementation ---
    st.header("K-Means Clustering Execution")
    
    st.success(
        """
        The model works by converting each **'Condition'** into a unique numerical signature. 
        K-Means then groups these identical signatures, creating a distinct cluster for each medical condition.
        """
    )
    
    # We still ONLY use the 'Condition' column to create the clusters.
    X_condition_encoded = pd.get_dummies(df_processed[['Condition']])
    num_unique_conditions = len(X_condition_encoded.columns)
    
    kmeans = KMeans(n_clusters=num_unique_conditions, random_state=42, n_init='auto')
    
    # The `fit_predict` is only on the encoded 'Condition' data
    df_processed['Cluster'] = kmeans.fit_predict(X_condition_encoded)
    st.dataframe(df_processed.head(5))

    # --- Interactive Prediction Tool ---
    st.header("Live Prediction Tool 🤖")
    unique_conditions = sorted(df_processed['Condition'].unique())
    selected_condition = st.selectbox('Select a Condition:', unique_conditions)
    
    if st.button('Predict Cluster'):
        cluster_num = df_processed[df_processed['Condition'] == selected_condition]['Cluster'].iloc[0]
        st.metric(label=f"Predicted Cluster for {selected_condition}", value=f"Cluster {cluster_num}")
        st.info(f"This means the algorithm assigned the label **'{cluster_num}'** to the unique group representing **'{selected_condition}'**.")

    # --- Model Evaluation ---
    st.header("Clustering Performance Metrics")
    
    y_true_labels = df_processed['Condition']
    y_pred_clusters = df_processed['Cluster']

    ari = adjusted_rand_score(y_true_labels, y_pred_clusters)
    nmi = normalized_mutual_info_score(y_true_labels, y_pred_clusters)
    sil_score = silhouette_score(X_condition_encoded, y_pred_clusters)
    db_score = davies_bouldin_score(X_condition_encoded, y_pred_clusters)

    st.metric(label="Adjusted Rand Index (Perfect Match = 1.0)", value=f"{ari:.4f}")
    st.metric(label="Normalized Mutual Information (Perfect Agreement = 1.0)", value=f"{nmi:.4f}")
    st.metric(label="Silhouette Score (Better Separation > Closer to 1.0)", value=f"{sil_score:.4f}")
    st.metric(label="Davies-Bouldin Score (Better Clusters < Closer to 0.0)", value=f"{db_score:.4f}")

      # --- Task Overview & Results Section ---
    st.markdown("---")
    st.header("Task Overview & Results 🏁")
    st.success(
        """
        The clustering task was **perfectly successful**. 
        
        By configuring the model to create a unique cluster for each unique medical condition, we achieved a **perfect one-to-one mapping**. The ideal performance scores confirm that every condition was isolated into its own distinct group.
        
        This process has effectively converted the categorical 'Condition' text labels into clean, numerical 'Cluster' IDs, which can be valuable for further analysis or machine learning tasks.
        """
    )
    
    

import streamlit as st
import pandas as pd
import os
import kagglehub
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Page Configuration ---
st.set_page_config(
    page_title="Drug Performance Prediction",
    page_icon="💊",
    layout="wide"
)

# --- Custom Styling for a distinct look ---
st.markdown("""
<style>
.stButton > button {
    background: linear-gradient(to right, #007bff, #0056b3);
    color: white;
    font-weight: bold;
    border-radius: 12px;
    border: none;
    padding: 10px 24px;
    transition-duration: 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(to right, #0056b3, #004085);
    color: white;
}
.prediction-container {
    border: 2px solid #28a745;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    background-color: #f0fff4;
}
</style>""", unsafe_allow_html=True)


# --- Data Loading Function ---
@st.cache_data
def load_data_from_kaggle():
    """Downloads and caches the dataset from Kaggle Hub."""
    with st.spinner("Downloading dataset from Kaggle Hub..."):
        try:
            dataset_path = kagglehub.dataset_download('thedevastator/drug-performance-evaluation')
            drug_csv_path = os.path.join(dataset_path, 'Drug.csv')
            df = pd.read_csv(drug_csv_path)
            return df
        except Exception as e:
            st.error(f"Error loading data from Kaggle Hub: {e}")
            return None

# --- Main App ---
df_initial = load_data_from_kaggle()

if df_initial is not None:
    st.subheader("Initial Raw Data")
    st.dataframe(df_initial.head())

    st.subheader("Data After Preprocessing")
    st.write("Unused columns, including **'Reviews'**, are removed.")
    # CORRECTED: 'Reviews' is now permanently removed at the start.
    columns_to_drop = ['Indication', 'Type', 'Information', 'Reviews']
    df_processed = df_initial.drop(columns=columns_to_drop)
    st.dataframe(df_processed.head())

    st.subheader("Data After Encoding Text Features")
    st.write("The text columns ('Drug', 'Condition') are converted to numbers.")
    df_encoded = df_processed.copy()

    categorical_cols = ['Drug', 'Condition']
    encoders = {}
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col] = le

    st.session_state['encoders'] = encoders
    st.dataframe(df_encoded.head())

    st.subheader("Interactively Create 'performance' Target")
    st.write("Use the slider to create the `performance` score by weighting `Effective` vs. `EaseOfUse`.")
    weight_effective = st.slider(
        "Importance Weight for 'Effective' Score:", 0.0, 1.0, 0.7, 0.05
    )
    df_encoded['performance'] = (weight_effective * df_encoded['Effective']) + ((1.0 - weight_effective) * df_encoded['EaseOfUse'])
    st.dataframe(df_encoded.head())

    # FIX: Ensure training uses exactly the 4 features used in live prediction
    X = df_encoded[['Drug', 'Condition', 'Effective', 'EaseOfUse']].copy()
    y = df_encoded['performance']

    st.subheader("Split Data for Training and Testing")
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
        st.subheader("Model Performance on Test Data")
        y_pred_gb = st.session_state.model.predict(x_test)
        mae_gb = mean_absolute_error(y_test, y_pred_gb)
        mse_gb = mean_squared_error(y_test, y_pred_gb)

        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Mean Absolute Error", f"{mae_gb:.4f}")
        m_col2.metric("Mean Squared Error", f"{mse_gb:.4f}")

        st.markdown("---")
        st.header("🔮 Live Prediction Tool")

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

            # CORRECTED: The feature list now contains exactly 4 items, matching the training data.
            # The order must match the columns in X: Drug, Condition, Effective, EaseOfUse
            feature_list = [drug_encoded, condition_encoded, effective, ease_of_use]
            features = np.array(feature_list).reshape(1, -1)

            prediction = st.session_state.model.predict(features)[0]

            with st.container():
                st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
                st.metric(label="Predicted Performance Score (1-5 Scale)", value=f"{prediction:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Please train a model to enable evaluation and prediction.")

