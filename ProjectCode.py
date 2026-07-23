# Import Libraries
import pandas as pd
import numpy as np
import joblib


# Load the dataset
df = pd.read_csv('Bachelor_Degree_Recommendation_Dataset_finalversion10.csv')
print("Dataset loaded successfully!")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Divide dataset into features and target variable
X = df.drop("Recommended_Department", axis=1)
y = df["Recommended_Department"]
print(y.value_counts())

# --- ONE-HOT ENCODE the categorical feature ---
# Intermediate_Group has no natural order, so one-hot encoding is correct here
X = pd.get_dummies(X, columns=["Intermediate_Group"], drop_first=False)
# Save the training column names
joblib.dump(X.columns.tolist(), "model_columns.joblib")

# --- LABEL ENCODE the target ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train test split (stratify keeps class proportions balanced in both sets)
train_X, test_X, train_y, test_y = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_X, train_y)
print("Model trained successfully!")

# --- Evaluate ---
train_acc = accuracy_score(train_y, model.predict(train_X))
test_acc = accuracy_score(test_y, model.predict(test_X))

print(f"\nTrain Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")

# Per-class report (convert back to original department names for readability)
test_y_labels = label_encoder.inverse_transform(test_y)
pred_labels = label_encoder.inverse_transform(model.predict(test_X))

print("\nClassification Report:\n")
print(classification_report(test_y_labels, pred_labels, zero_division=0))

joblib.dump(model, 'bachelor_degree_recommendation_model.joblib')
# Save the label encoder
joblib.dump(label_encoder, "label_encoder.joblib")
