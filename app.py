import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="FraudGuard AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Global FinTech UI Theme
# -------------------------------------------------
st.markdown("""
<style>

/* App background */
.stApp {
    background: linear-gradient(135deg, #0E2F24 0%, #1B4D3E 60%, #2E6B57 100%);
    color: #F5F5F2;
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0B241C;
    padding-top: 2rem;
}

/* Headings – Editorial feel */
h1, h2 {
    font-family: 'Playfair Display', serif;
    letter-spacing: -0.5px;
}
h3 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
}

/* Hero text spacing */
.block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
}

/* Metric cards – clean, premium */
.metric-card {
    background-color: #F7F7F4;
    color: #1C2E26;
    padding: 28px;
    border-radius: 20px;
    box-shadow: none;
    border: 1px solid rgba(0,0,0,0.05);
    text-align: center;
}

/* Buttons – Ellevest style */
.stButton>button {
    background-color: #E6EEDC;
    color: #1B3A2E;
    border-radius: 999px;
    font-weight: 600;
    padding: 0.7rem 2rem;
    border: none;
}
.stButton>button:hover {
    background-color: #D8E4CB;
}

/* Inputs */
input {
    border-radius: 999px !important;
    padding: 0.6rem 1rem !important;
}

/* Dataframes */
.stDataFrame {
    background-color: #FFFFFF;
    border-radius: 16px;
    border: none;
}

/* Remove default Streamlit clutter */
footer {visibility: hidden;}
header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Sidebar Branding
# -------------------------------------------------
st.sidebar.markdown("""
<h2 style="color:#A7E6C5;">FraudGuard AI</h2>
<p style="font-size:13px;opacity:0.85;">
AI-powered Financial Fraud Protection Platform
</p>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "📊 Data Analysis", "🤖 Model Training", "🔮 Predict Fraud"]
)

# -------------------------------------------------
# Hero Section
# -------------------------------------------------
st.markdown("""
<div style="max-width:650px;padding:80px 0;">
    <h1 style="font-size:52px;line-height:1.1;">
        Your financial security<br>is personal
    </h1>
    <p style="font-size:18px;opacity:0.9;margin-top:16px;">
        FraudGuard AI delivers intelligent, personalized protection
        for modern financial transactions.
    </p>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------
# Load Data
# -------------------------------------------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# -------------------------------------------------
# Preprocessing
# -------------------------------------------------
def preprocess_data(df, balance="none"):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if balance == "smote":
        X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
    elif balance == "under":
        X_train, y_train = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, scaler

# -------------------------------------------------
# Model Training
# -------------------------------------------------
def train_model(X, y, model_type):
    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    model.fit(X, y)
    return model

# -------------------------------------------------
# Evaluation
# -------------------------------------------------
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "cm": confusion_matrix(y_test, y_pred),
        "prob": y_prob
    }

# -------------------------------------------------
# Dashboard Page
# -------------------------------------------------
if page == "🏠 Dashboard":

    if "data" not in st.session_state:
        st.info("Upload a dataset in Data Analysis section.")
    else:
        df = st.session_state["data"]
        fraud_count = df["Class"].sum()
        fraud_rate = (fraud_count / len(df)) * 100

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"<div class='metric-card'><h3>Total Transactions</h3><h2>{len(df):,}</h2></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-card'><h3>Fraud Cases</h3><h2>{fraud_count}</h2></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='metric-card'><h3>Fraud Rate</h3><h2>{fraud_rate:.2f}%</h2></div>", unsafe_allow_html=True)
        with c4:
            status = "Trained" if "model" in st.session_state else "Not Trained"
            st.markdown(f"<div class='metric-card'><h3>Model Status</h3><h2>{status}</h2></div>", unsafe_allow_html=True)

        st.markdown("### Transaction Distribution")
        fig, ax = plt.subplots()
        ax.bar(["Normal", "Fraud"], df["Class"].value_counts())
        ax.set_facecolor("#F7FAF8")
        fig.patch.set_facecolor("#F7FAF8")
        st.pyplot(fig)

# -------------------------------------------------
# Data Analysis Page
# -------------------------------------------------
elif page == "📊 Data Analysis":

    file = st.file_uploader("Upload Credit Card Dataset (CSV)", type="csv")

    if file:
        df = load_data(file)
        st.session_state["data"] = df

        t1, t2, t3 = st.tabs(["Preview", "Statistics", "Correlation"])

        with t1:
            st.dataframe(df.head(10))
        with t2:
            st.dataframe(df.describe())
        with t3:
            corr = df.corr()["Class"].sort_values(ascending=False)
            st.bar_chart(corr[1:11])

# -------------------------------------------------
# Model Training Page
# -------------------------------------------------
elif page == "🤖 Model Training":

    if "data" not in st.session_state:
        st.warning("Upload data first.")
    else:
        model_type = st.selectbox("Select Model", ["Logistic Regression", "Random Forest"])
        balance = st.selectbox("Class Balancing", ["None", "SMOTE", "Under-sampling"])

        if st.button("Train Model"):
            X_train, X_test, y_train, y_test, scaler = preprocess_data(
                st.session_state["data"],
                "smote" if balance=="SMOTE" else "under" if balance=="Under-sampling" else "none"
            )

            model = train_model(X_train, y_train, "logistic" if model_type=="Logistic Regression" else "rf")

            metrics = evaluate(model, X_test, y_test)

            st.session_state["model"] = model
            st.session_state["scaler"] = scaler
            st.session_state["features"] = st.session_state["data"].drop("Class", axis=1).columns.tolist()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            c2.metric("Precision", f"{metrics['precision']:.3f}")
            c3.metric("Recall", f"{metrics['recall']:.3f}")
            c4.metric("F1 Score", f"{metrics['f1']:.3f}")

# -------------------------------------------------
# Prediction Page
# -------------------------------------------------
if pred == 1:
    st.markdown(
        f"""
        <div class="metric-card" style="border-left:6px solid #9F2D2D;">
            <h3>Potential Fraud Identified</h3>
            <p>Confidence level: {prob*100:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f"""
        <div class="metric-card" style="border-left:6px solid #2F7D64;">
            <h3>Transaction Appears Safe</h3>
            <p>Confidence level: {prob*100:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------------------------------------
# Footer
# -------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.info("""
**Educational & Demo Project**  
FraudGuard AI – ML-based FinTech Dashboard  

Technologies: Streamlit, Scikit-learn, Pandas, NumPy
""")
