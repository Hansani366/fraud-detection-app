"""
Credit Card Fraud Detection Application
A comprehensive Streamlit app for detecting fraudulent credit card transactions
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Model Training", "Predict Fraud"])

# Function to load data
@st.cache_data
def load_data(file):
    """Load and return the dataset"""
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to preprocess data
def preprocess_data(df, balance_method='none'):
    """
    Preprocess the data: scaling and train-test split
    
    Parameters:
    - df: DataFrame with features and target
    - balance_method: 'smote', 'undersample', or 'none'
    """
    # Separate features and target
    if 'Class' not in df.columns:
        st.error("Dataset must contain a 'Class' column indicating fraud (1) or normal (0)")
        return None, None, None, None, None
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance
    if balance_method == 'smote':
        st.info("Applying SMOTE to balance classes...")
        smote = SMOTE(random_state=42)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
    elif balance_method == 'undersample':
        st.info("Applying Random Under-sampling to balance classes...")
        rus = RandomUnderSampler(random_state=42)
        X_train_scaled, y_train = rus.fit_resample(X_train_scaled, y_train)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Function to train model
def train_model(X_train, y_train, model_type='logistic'):
    """Train and return the model"""
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:  # random_forest
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    with st.spinner(f'Training {model_type} model...'):
        model.fit(X_train, y_train)
    
    return model

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
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

# Function to plot confusion matrix
def plot_confusion_matrix(cm):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    ax.set_xticklabels(['Normal', 'Fraud'])
    ax.set_yticklabels(['Normal', 'Fraud'])
    return fig

# Function to plot ROC curve
def plot_roc_curve(y_test, y_pred_proba):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return fig

# HOME PAGE
if page == "Home":
    st.write("## Welcome to the Credit Card Fraud Detection System")
    
    st.write("""
    This application uses Machine Learning to detect fraudulent credit card transactions.
    
    ### How it works:
    1. **Upload Data**: Load your transaction dataset (CSV format)
    2. **Analyze**: Explore the data distribution and statistics
    3. **Train Model**: Choose and train a machine learning model
    4. **Predict**: Detect fraud in new transactions
    
    ### Dataset Requirements:
    - CSV format with numerical features
    - Must contain a 'Class' column (0 = Normal, 1 = Fraud)
    - Common dataset: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
    
    ### Features:
    - ✅ Multiple ML algorithms (Logistic Regression, Random Forest)
    - ✅ Handle imbalanced data (SMOTE, Under-sampling)
    - ✅ Comprehensive evaluation metrics
    - ✅ Interactive visualizations
    - ✅ Real-time predictions
    """)
    
    st.info("👈 Use the sidebar to navigate through different sections")

# DATA ANALYSIS PAGE
elif page == "Data Analysis":
    st.write("## 📊 Data Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)
        
        if df is not None:
            # Store in session state
            st.session_state['data'] = df
            
            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
            
            # Display basic info
            st.write("### Dataset Preview")
            st.dataframe(df.head(10))
            
            # Display statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Transactions", len(df))
            with col2:
                if 'Class' in df.columns:
                    fraud_count = df['Class'].sum()
                    st.metric("Fraudulent Transactions", fraud_count)
            with col3:
                if 'Class' in df.columns:
                    fraud_percentage = (df['Class'].sum() / len(df)) * 100
                    st.metric("Fraud Percentage", f"{fraud_percentage:.2f}%")
            
            # Class distribution
            if 'Class' in df.columns:
                st.write("### Class Distribution")
                fig, ax = plt.subplots(figsize=(10, 5))
                
                class_counts = df['Class'].value_counts()
                ax.bar(['Normal', 'Fraud'], class_counts.values, color=['#2ecc71', '#e74c3c'])
                ax.set_ylabel('Count')
                ax.set_title('Transaction Class Distribution')
                ax.grid(axis='y', alpha=0.3)
                
                for i, v in enumerate(class_counts.values):
                    ax.text(i, v + 100, str(v), ha='center', va='bottom')
                
                st.pyplot(fig)
                
                # Feature statistics
                st.write("### Feature Statistics")
                st.dataframe(df.describe())
                
                # Correlation with target
                st.write("### Feature Correlation with Fraud")
                if len(df.columns) > 2:
                    correlations = df.corr()['Class'].sort_values(ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    correlations[1:11].plot(kind='barh', ax=ax, color='steelblue')
                    ax.set_xlabel('Correlation Coefficient')
                    ax.set_title('Top 10 Features Correlated with Fraud')
                    ax.grid(axis='x', alpha=0.3)
                    st.pyplot(fig)

# MODEL TRAINING PAGE
elif page == "Model Training":
    st.write("## 🤖 Model Training")
    
    if 'data' not in st.session_state:
        st.warning("⚠️ Please upload data in the 'Data Analysis' section first!")
    else:
        df = st.session_state['data']
        
        st.write("### Configure Training Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select Model",
                ["logistic", "random_forest"],
                format_func=lambda x: "Logistic Regression" if x == "logistic" else "Random Forest"
            )
        
        with col2:
            balance_method = st.selectbox(
                "Class Balancing Method",
                ["none", "smote", "undersample"],
                format_func=lambda x: x.upper() if x != "none" else "None"
            )
        
        if st.button("Train Model", type="primary"):
            # Preprocess data
            X_train, X_test, y_train, y_test, scaler = preprocess_data(df, balance_method)
            
            if X_train is not None:
                # Display class distribution after balancing
                st.write("#### Training Set Class Distribution")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Normal Transactions", int((y_train == 0).sum()))
                with col2:
                    st.metric("Fraud Transactions", int((y_train == 1).sum()))
                
                # Train model
                model = train_model(X_train, y_train, model_type)
                
                # Evaluate model
                metrics = evaluate_model(model, X_test, y_test)
                
                # Store model and scaler in session state
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['feature_names'] = df.drop('Class', axis=1).columns.tolist()
                
                st.success("✅ Model trained successfully!")
                
                # Display metrics
                st.write("### 📈 Model Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.4f}")
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.4f}")
                with col4:
                    st.metric("F1-Score", f"{metrics['f1']:.4f}")
                
                # Confusion Matrix
                st.write("### Confusion Matrix")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = plot_confusion_matrix(metrics['confusion_matrix'])
                    st.pyplot(fig)
                
                with col2:
                    st.write("#### Interpretation:")
                    tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
                    st.write(f"- **True Negatives (TN)**: {tn}")
                    st.write(f"- **False Positives (FP)**: {fp}")
                    st.write(f"- **False Negatives (FN)**: {fn}")
                    st.write(f"- **True Positives (TP)**: {tp}")
                    st.write(f"\n**Fraud Detection Rate**: {tp/(tp+fn)*100:.2f}%")
                
                # ROC Curve
                st.write("### ROC Curve")
                fig = plot_roc_curve(y_test, metrics['y_pred_proba'])
                st.pyplot(fig)
                
                # Classification Report
                st.write("### Detailed Classification Report")
                report = classification_report(y_test, metrics['y_pred'], 
                                              target_names=['Normal', 'Fraud'],
                                              output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

# PREDICTION PAGE
elif page == "Predict Fraud":
    st.write("## 🔍 Predict Fraud")
    
    if 'model' not in st.session_state:
        st.warning("⚠️ Please train a model in the 'Model Training' section first!")
    else:
        st.write("### Enter Transaction Details")
        
        model = st.session_state['model']
        scaler = st.session_state['scaler']
        feature_names = st.session_state['feature_names']
        
        # Create input method tabs
        tab1, tab2 = st.tabs(["Manual Input", "Batch Prediction"])
        
        with tab1:
            st.write("Enter feature values for the transaction:")
            
            # Create input fields dynamically
            input_data = {}
            cols = st.columns(3)
            
            for idx, feature in enumerate(feature_names):
                with cols[idx % 3]:
                    input_data[feature] = st.number_input(
                        f"{feature}",
                        value=0.0,
                        format="%.6f",
                        key=f"input_{feature}"
                    )
            
            if st.button("Predict", type="primary"):
                # Prepare input
                input_df = pd.DataFrame([input_data])
                input_scaled = scaler.transform(input_df)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                prediction_proba = model.predict_proba(input_scaled)[0]
                
                # Display result
                st.write("### Prediction Result")
                
                if prediction == 1:
                    st.error("🚨 **FRAUD DETECTED!**")
                    st.write(f"Fraud Probability: **{prediction_proba[1]*100:.2f}%**")
                else:
                    st.success("✅ **NORMAL TRANSACTION**")
                    st.write(f"Fraud Probability: **{prediction_proba[1]*100:.2f}%**")
                
                # Probability bar chart
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(['Normal', 'Fraud'], prediction_proba, color=['#2ecc71', '#e74c3c'])
                ax.set_xlabel('Probability')
                ax.set_title('Prediction Probabilities')
                ax.set_xlim([0, 1])
                
                for i, v in enumerate(prediction_proba):
                    ax.text(v + 0.02, i, f'{v*100:.2f}%', va='center')
                
                st.pyplot(fig)
        
        with tab2:
            st.write("Upload a CSV file with transactions to predict:")
            uploaded_file = st.file_uploader("Upload CSV for batch prediction", type=['csv'])
            
            if uploaded_file is not None:
                batch_df = pd.read_csv(uploaded_file)
                st.write("#### Data Preview")
                st.dataframe(batch_df.head())
                
                if st.button("Predict Batch", type="primary"):
                    # Ensure all features are present
                    missing_features = set(feature_names) - set(batch_df.columns)
                    if missing_features:
                        st.error(f"Missing features: {missing_features}")
                    else:
                        # Scale and predict
                        batch_scaled = scaler.transform(batch_df[feature_names])
                        predictions = model.predict(batch_scaled)
                        predictions_proba = model.predict_proba(batch_scaled)[:, 1]
                        
                        # Add predictions to dataframe
                        batch_df['Prediction'] = predictions
                        batch_df['Fraud_Probability'] = predictions_proba
                        batch_df['Status'] = batch_df['Prediction'].map({0: 'Normal', 1: 'Fraud'})
                        
                        st.write("### Prediction Results")
                        st.dataframe(batch_df)
                        
                        # Summary
                        fraud_count = predictions.sum()
                        st.write(f"**Total Fraudulent Transactions Detected**: {fraud_count} out of {len(batch_df)}")
                        
                        # Download results
                        csv = batch_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="fraud_predictions.csv",
                            mime="text/csv"
                        )

st.sidebar.markdown("---")
st.sidebar.info("""
**Developed for Educational Purposes**

This application demonstrates ML-based credit card fraud detection using Python and Streamlit.

**Technologies Used:**
- Streamlit
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Imbalanced-learn
""")


