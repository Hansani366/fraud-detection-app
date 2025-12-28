import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

# ---------------------- Page Config ----------------------
st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- Dark/Light Theme ----------------------
theme = st.sidebar.radio("🌗 Theme", ["Light","Dark"])

if theme == "Dark":
    st.markdown("""
    <style>
    body {background-color: #0E1117; color: #FFFFFF;}
    .main-header {color: #1AB1FF;}
    .stButton>button {background-color: #1AB1FF; color: #FFFFFF;}
    .stMetric {background-color: #1B1F28; border-radius: 12px; padding: 10px;}
    .stDataFrame {color: #FFFFFF; background-color: #0E1117;}
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .main-header {color: #1f77b4;}
    .stMetric {background-color: #FFFFFF; border-radius: 12px; padding: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);}
    </style>
    """, unsafe_allow_html=True)

# ---------------------- Header ----------------------
st.markdown('<h1 class="main-header" style="text-align:center;font-family:sans-serif;">💳 FraudGuard AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:gray;font-family:sans-serif;">AI-powered Credit Card Fraud Detection Dashboard</p>', unsafe_allow_html=True)

# ---------------------- Sidebar Navigation ----------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Dashboard", "📊 Data Analysis", "🤖 Model Training", "🔮 Predict Fraud"]
)

# ---------------------- Load Data ----------------------
@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ---------------------- Preprocess ----------------------
def preprocess_data(df, balance_method='none'):
    if 'Class' not in df.columns:
        st.error("Dataset must contain a 'Class' column (0=Normal,1=Fraud)")
        return None, None, None, None, None

    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if balance_method=='smote':
        smote = SMOTE(random_state=42)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
    elif balance_method=='undersample':
        rus = RandomUnderSampler(random_state=42)
        X_train_scaled, y_train = rus.fit_resample(X_train_scaled, y_train)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# ---------------------- Model ----------------------
def train_model(X_train, y_train, model_type='logistic'):
    if model_type=='logistic':
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    with st.spinner(f"Training {model_type} model..."):
        model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:,1]
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    return metrics

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    return fig

def plot_roc_curve(y_test, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(fpr, tpr, color='#1AB1FF', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0,1],[0,1], color='gray', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
    return fig

# ---------------------- DASHBOARD ----------------------
if page=="🏠 Dashboard":
    st.markdown("## 📊 Dashboard Overview")
    if 'data' in st.session_state:
        df = st.session_state['data']
        fraud_count = df['Class'].sum()
        fraud_percentage = (fraud_count/len(df))*100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📈 Total Transactions", f"{len(df):,}")
        col2.metric("🚨 Fraud Cases", fraud_count)
        col3.metric("⚡ Fraud Rate", f"{fraud_percentage:.2f}%")
        col4.metric("🤖 Model Status", "Trained ✅" if 'model' in st.session_state else "Not Trained ❌")

        st.markdown("### Class Distribution")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(['Normal','Fraud'], df['Class'].value_counts().values, color=['#2ecc71','#e74c3c'])
        st.pyplot(fig)
    else:
        st.info("Upload dataset first in 📊 Data Analysis.")

# ---------------------- DATA ANALYSIS ----------------------
elif page=="📊 Data Analysis":
    st.markdown("## Analyze Your Data")
    uploaded_file = st.file_uploader("Upload CSV dataset", type=['csv'])
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state['data'] = df
            tabs = st.tabs(["📄 Preview","📈 Stats","🔗 Correlation"])
            with tabs[0]: st.dataframe(df.head(10))
            with tabs[1]: st.dataframe(df.describe())
            with tabs[2]:
                corr = df.corr()['Class'].sort_values(ascending=False)
                st.bar_chart(corr[1:11])

# ---------------------- MODEL TRAINING ----------------------
elif page=="🤖 Model Training":
    st.markdown("## Train Your Model")
    if 'data' not in st.session_state:
        st.warning("Upload data first in 📊 Data Analysis")
    else:
        df = st.session_state['data']
        model_type = st.selectbox("Select Model", ["Logistic Regression", "Random Forest"])
        balance_method = st.selectbox("Class Balancing", ["None","SMOTE","Under-sampling"])
        if st.button("Train Model"):
            X_train, X_test, y_train, y_test, scaler = preprocess_data(
                df, balance_method.lower() if balance_method!="None" else 'none'
            )
            model = train_model(X_train, y_train, 'logistic' if model_type=="Logistic Regression" else "random_forest")
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['feature_names'] = df.drop('Class',axis=1).columns.tolist()
            metrics = evaluate_model(model, X_test, y_test)
            st.success("✅ Model Trained")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            col2.metric("Precision", f"{metrics['precision']:.4f}")
            col3.metric("Recall", f"{metrics['recall']:.4f}")
            col4.metric("F1-Score", f"{metrics['f1']:.4f}")
            st.pyplot(plot_confusion_matrix(metrics['confusion_matrix']))
            st.pyplot(plot_roc_curve(y_test, metrics['y_pred_proba']))

# ---------------------- PREDICTION ----------------------
elif page=="🔮 Predict Fraud":
    st.markdown("## Predict Fraud Transactions")
    if 'model' not in st.session_state:
        st.warning("Train a model first in 🤖 Model Training")
    else:
        model = st.session_state['model']
        scaler = st.session_state['scaler']
        features = st.session_state['feature_names']
        tab1, tab2 = st.tabs(["Manual Input","Batch Prediction"])

        with tab1:
            st.subheader("Manual Transaction Prediction")
            input_data = {f: st.number_input(f"{f}", 0.0) for f in features}
            if st.button("Predict Transaction"):
                input_scaled = scaler.transform(pd.DataFrame([input_data]))
                pred = model.predict(input_scaled)[0]
                prob = model.predict_proba(input_scaled)[0]
                if pred==1:
                    st.error(f"🚨 FRAUD DETECTED! ({prob[1]*100:.2f}%)")
                else:
                    st.success(f"✅ NORMAL TRANSACTION ({prob[1]*100:.2f}%)")

        with tab2:
            st.subheader("Batch Prediction (CSV Upload)")
            file = st.file_uploader("CSV File", type=['csv'], key='batch')
            if file:
                batch_df = pd.read_csv(file)
                missing = set(features)-set(batch_df.columns)
                if missing: st.error(f"Missing Features: {missing}")
                else:
                    scaled = scaler.transform(batch_df[features])
                    batch_df['Prediction'] = model.predict(scaled)
                    batch_df['Fraud_Probability'] = model.predict_proba(scaled)[:,1]
                    batch_df['Status'] = batch_df['Prediction'].map({0:'Normal',1:'Fraud'})
                    st.dataframe(batch_df)
                    st.download_button("Download CSV", batch_df.to_csv(index=False), file_name="fraud_predictions.csv")

# ---------------------- Sidebar Info ----------------------
st.sidebar.markdown("---")
st.sidebar.info("""
**Developed for Educational & Demo Purposes**  
AI-powered Credit Card Fraud Detection using Python & Streamlit  

**Technologies:** Streamlit, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
""")
