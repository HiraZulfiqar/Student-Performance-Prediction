# Student Performance Prediction

A Machine Learning regression project that predicts students' final scores and compares different regression models, including Linear Regression, Ridge, Lasso, and ElasticNet.

## 📌 Project Overview

This project uses student performance data to predict the `Final_Score` based on the available student features.

The project also demonstrates **Regularization**, a technique used to reduce model overfitting and improve generalization.

## 🚀 What This Project Does

* Loads and explores the student performance dataset
* Checks dataset shape, columns, data types, and missing values
* Separates features (`X`) and target (`y`)
* Splits data into training and testing sets
* Standardizes numerical features
* Trains a Linear Regression model
* Trains Ridge Regression
* Trains Lasso Regression
* Trains ElasticNet Regression
* Evaluates models using RMSE and R² Score
* Compares model performance
* Compares model coefficients
* Visualizes actual vs predicted scores

## 🧠 Machine Learning Workflow

```text
Student Dataset
      ↓
Data Understanding
      ↓
Feature / Target Separation
      ↓
Train-Test Split
      ↓
Feature Scaling
      ↓
Linear Regression
      ↓
Regularization
 ┌────┼─────┐
 ↓    ↓     ↓
Ridge Lasso ElasticNet
 └────┼─────┘
      ↓
Model Comparison
      ↓
Prediction
      ↓
Visualization
```

## 🤖 Models Used

### 1. Linear Regression

A basic regression model used as the baseline model.

### 2. Ridge Regression

Uses **L2 regularization** to reduce the effect of large coefficients and help control overfitting.

### 3. Lasso Regression

Uses **L1 regularization** and can reduce some feature coefficients toward zero.

### 4. ElasticNet

Combines **L1 and L2 regularization**, giving it properties of both Lasso and Ridge.

## 📊 Evaluation Metrics

### RMSE

**Root Mean Squared Error** measures how far predictions are from the actual values.

Lower RMSE is better.

### R² Score

Measures how well the model explains the variation in the target.

Higher R² is generally better.

## 📈 Visualization

The project creates an **Actual vs Predicted** scatter plot for the ElasticNet model.

A prediction closer to the diagonal line indicates better predictions.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

## 📁 Project Structure

```text
Student-Performance-Prediction/
│
├── student_performance.csv
├── main.py
├── README.md
└── requirements.txt
```

## ▶️ How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python main.py
```

## 🎯 Learning Goals

This project was created to practice:

* Regression
* Train/Test Split
* Feature Scaling
* Linear Regression
* Regularization
* Ridge Regression
* Lasso Regression
* ElasticNet
* Model Evaluation
* RMSE
* R² Score
* Model Comparison
* Data Visualization

## 🔑 Key Takeaway

The main goal of this project is to understand how **regularization techniques such as Ridge, Lasso, and ElasticNet** can be used alongside Linear Regression to build and compare regression models.
