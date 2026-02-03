# Loan Approval Prediction Using KNN

## 📌 Project Overview
This project focuses on predicting whether a loan will be approved based on applicant details using Machine Learning.  
It applies data preprocessing techniques and the K-Nearest Neighbors (KNN) algorithm to build a reliable classification model.

## 🛠 Technologies Used
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Scikit-learn  

## 📂 Dataset
The dataset contains applicant information such as:

- Gender  
- Marital Status  
- Dependents  
- Education  
- Self Employment  
- Applicant Income  
- Loan Amount  
- Credit History  
- Loan Status

  
## 🔍 Project Workflow

1. Data loading and inspection  
2. Handling missing values using mode and median  
3. One-hot encoding of categorical variables  
4. Feature scaling using StandardScaler  
5. Train-test split  
6. KNN model training  
7. Model evaluation using accuracy and confusion matrix  
8. Cross-validation  
9. Hyperparameter tuning using Elbow Method  

## 📈 Elbow Method for Optimal K Value

This graph shows the error rate for different K values:
<img width="977" height="665" alt="knn error rate  image " src="https://github.com/user-attachments/assets/858d2a07-b895-4885-915b-588d6f69a8e4" />

## ✅ Model Performance

- Training Accuracy: (add your value)  
- Testing Accuracy: (add your value)  
- Cross Validation Accuracy: (add your value)  

## 📊 Results

The optimized KNN classifier provides good predictive accuracy and generalization using cross-validation.


## 🚀 How to Run the Project

```bash
pip install pandas numpy matplotlib scikit-learn
python loan_prediction.py
