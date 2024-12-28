import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import json

# Load data
data = pd.read_csv('classification_model_data.csv')
print("Data loaded successfully:")
print(data.head(5))

# Separate features and target
features = data.drop(columns=['activity_class'])
targets = data.activity_class
print("\nFeatures and targets separated.")
print("Feature data preview:")
print(features.head(5))
print("Target data preview:")
print(targets.head(5))

# Remove low-variance features
def remove_low_variance(input_data, threshold=0.1):
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]]

features = remove_low_variance(features, threshold=0.1)
print("\nFeatures after removing low variance:")
print(features.head(5))

# Metrics function
def get_metrics(predicted, true):
    metrics = dict()
    metrics['accuracy'] = round(accuracy_score(predicted, true), 5)
    metrics['precision'] = round(precision_score(predicted, true, average='weighted'), 5)
    metrics['recall'] = round(recall_score(predicted, true, average='weighted'), 5)
    metrics['f1'] = round(f1_score(predicted, true, average='weighted'), 5)

    return metrics

# Split data
X_training_set, X_validation_set, y_training_set, y_validation_set = train_test_split(features, targets, test_size=0.2, random_state=42)
print("\nData split into training and validation sets.")
print(f"Training set size: {X_training_set.shape}, Validation set size: {X_validation_set.shape}")

# Train Random Forest Classifier
RF_model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
RF_model.fit(X_training_set, y_training_set)
print("\nRandom Forest model trained.")

# Save the trained Random Forest model for later use
joblib.dump(RF_model, 'trained_model.joblib')  # Saves model to 'trained_model.joblib'
print("Random Forest model saved as 'trained_model.joblib'.")

# Predictions and Metrics for Random Forest
y_training_pred = RF_model.predict(X_training_set)
y_validation_pred = RF_model.predict(X_validation_set)
RF_mcc_test = matthews_corrcoef(y_validation_set, y_validation_pred)
print("\nRandom Forest MCC (Validation):", RF_mcc_test)

RF_metrics = pd.DataFrame([get_metrics(y_validation_pred, y_validation_set)])
print("\nRandom Forest validation metrics:")
print(RF_metrics)

# Train SVM Classifier
SVM_classifier = LinearSVC(max_iter=10000)
SVM_classifier.fit(X_training_set, y_training_set)
print("\nSVM model trained.")

# Predictions and Metrics for SVM
y_SVM_pred = SVM_classifier.predict(X_validation_set)
SVM_metrics = pd.DataFrame([get_metrics(y_SVM_pred, y_validation_set)])
print("\nSVM validation metrics:")
print(SVM_metrics)

# Save metrics
RF_metrics.to_csv("RF_metrics.csv", index=False)
SVM_metrics.to_csv("SVM_metrics.csv", index=False)
print("\nMetrics saved to RF_metrics.csv and SVM_metrics.csv")
