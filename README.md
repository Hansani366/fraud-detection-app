# 💳 Credit Card Fraud Detection System

A comprehensive **Streamlit-based Machine Learning web application** for detecting fraudulent credit card transactions.  
This project demonstrates how ML models can be used to identify fraud in highly imbalanced datasets with interactive analysis and real-time predictions.

---

## 🚀 Live Demo

🔗 **Deployed App:**  
https://fraud-detection-app-pddfjorx659qejjuvd8k4y.streamlit.app/

---

## 📌 Features

- 📊 Interactive data analysis and visualization
- 🤖 Machine Learning model training
  - Logistic Regression
  - Random Forest
- ⚖️ Class imbalance handling
  - SMOTE (Over-sampling)
  - Random Under-sampling
- 📈 Model evaluation metrics
  - Accuracy, Precision, Recall, F1-score
  - Confusion Matrix
  - ROC Curve
- 🔍 Fraud prediction
  - Manual single-transaction prediction
  - Batch prediction via CSV upload
- 📥 Download prediction results as CSV

---

## 🧠 Machine Learning Workflow

1. Upload credit card transaction dataset (CSV)
2. Perform exploratory data analysis
3. Preprocess data (scaling & balancing)
4. Train ML models
5. Evaluate performance
6. Predict fraud on new transactions

---

## 📂 Dataset Requirements

- CSV format
- Numerical features only
- Must contain a **`Class`** column:
  - `0` → Normal Transaction  
  - `1` → Fraudulent Transaction  

📌 Recommended Dataset:  
[Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

---

## 🛠️ Technologies Used

- **Frontend / Web App:** Streamlit
- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Imbalanced Data Handling:** imbalanced-learn

---
