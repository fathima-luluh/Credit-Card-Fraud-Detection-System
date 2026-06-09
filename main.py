import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

print("=" * 50)
print("CREDIT CARD FRAUD DETECTION SYSTEM")
print("=" * 50)

# Load dataset
print("\nLoading dataset...")

df = pd.read_csv("data/creditcard.csv")

print("Dataset Loaded Successfully!")

# Dataset overview
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fraud distribution
print("\nFraud Distribution:")
print(df['Class'].value_counts())

# Plot fraud distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)

plt.title("Fraud vs Normal Transactions")
plt.xlabel("Class")
plt.ylabel("Count")

plt.savefig("images/fraud_distribution.png")
plt.show()

# Features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Scale Amount feature
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Handle imbalance using SMOTE
print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nBefore SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

# Train model
print("\nTraining Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_smote, y_train_smote)

print("Model Training Completed!")

# Predictions
print("\nMaking Predictions...")

y_pred = model.predict(X_test)

# Evaluation
print("\nMODEL EVALUATION")
print("=" * 50)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("images/confusion_matrix.png")

plt.show()

# Feature importance
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop Important Features:")
print(feature_importance.head(10))

plt.figure(figsize=(10,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance.head(10)
)

plt.title("Top 10 Important Features")

plt.savefig("images/feature_importance.png")

plt.show()

# Sample prediction
print("\nSample Prediction:")

sample_transaction = X_test.iloc[0:1]

prediction = model.predict(sample_transaction)[0]

if prediction == 1:
    print("⚠️ Fraudulent Transaction Detected!")
else:
    print("✅ Normal Transaction")

print("\nProject Completed Successfully!")


import joblib

# Save trained model
joblib.dump(model, "models/fraud_model.pkl")

print("\nModel Saved Successfully!")