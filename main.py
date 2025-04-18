# Importing all the important libraries
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import RandomOverSampler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Reading the dataset
df = pd.read_csv("/kaggle/input/credit-card-fraud-detection-dataset-2023/creditcard_2023.csv")
df.head()

# Checking the size of the dataset
df.shape

# Checking the info of all columns
df.info()

# Checking if there are any missing values
df.isna().sum()

# Plotting correlation heatmap
plt.style.use("seaborn")
plt.rcParams['figure.figsize'] = (22, 11)
plt.title("Correlation Heatmap", fontsize=18, weight='bold')

sns.heatmap(df.corr(), cmap="BuPu", annot=True)
plt.show()

# Plotting the class distribution to see the imbalance
df['Class'].value_counts(normalize=True).plot(kind='bar')
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.title("Class Balance")
plt.show()

# Separating features and target
x = df.drop(['id', 'Class'], axis=1)
y = df['Class']

# Scaling the features
stn_scaler = StandardScaler()
x_scaled = stn_scaler.fit_transform(x)

# Putting back into a DataFrame
X = pd.DataFrame(x_scaled, columns=x.columns)

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Checking shapes
print("X shape:", X.shape)
print("y shape:", y.shape)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Calculating baseline accuracy (if we always predicted the majority class)
acc_baseline = y_train.value_counts(normalize=True).max()
print("Baseline Accuracy:", round(acc_baseline, 4))

# Creating the Logistic Regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Training and Testing Accuracy
acc_train = clf.score(X_train, y_train)
acc_test = clf.score(X_test, y_test)

print(f"Training accuracy: {round(acc_train, 4)}")
print(f"Test accuracy: {round(acc_test, 4)}")

# Showing the confusion matrix
ConfusionMatrixDisplay.from_estimator(
    clf,
    X_test,
    y_test
)
plt.show()

# Showing full classification report
print(classification_report(
    y_test,
    clf.predict(X_test)
))

# Checking which features were most important
features = X_test.columns
importances = clf.coef_[0]

# Plotting feature importance
feat_imp = pd.Series(importances, index=features).sort_values()
feat_imp.tail().plot(kind='barh')
plt.xlabel("Importance Scale")
plt.ylabel("Feature")
plt.title("Feature Importance")
plt.show()
