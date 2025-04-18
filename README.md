﻿# Credit Card Fraud Detection Project

This project is a simple machine learning pipeline to detect fraudulent credit card transactions.  
It uses a logistic regression model to classify transactions as fraud or not fraud.

The main steps included are:

- Importing libraries
- Loading the dataset
- Exploring the data (EDA) using info, missing values check, and correlation heatmap
- Visualizing the class imbalance
- Preprocessing the data (dropping unnecessary columns and scaling features)
- Splitting the data into training and testing sets
- Training a Logistic Regression model
- Evaluating the model using accuracy, confusion matrix, and classification report
- Plotting feature importance

## Dataset
The dataset used is from Kaggle:
- `Credit Card Fraud Detection Dataset 2023`
- File: `creditcard_2023.csv`

## Libraries Used
- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn
- imblearn

## How to Run
1. Make sure you have Python installed.
2. Install the required libraries:
   ```
   pip install numpy pandas seaborn matplotlib scikit-learn imbalanced-learn
   ```
3. Load the dataset into the same directory or change the path in the code.
4. Run the Python script.

## Output
- Correlation heatmap
- Class balance bar chart
- Confusion matrix
- Classification report (Precision, Recall, F1 Score)
- Feature importance bar chart

## Notes
- The data is highly imbalanced (very few fraud cases).
- Logistic Regression is used for simplicity, but you can try other models like Random Forest, XGBoost, etc. for better performance.

