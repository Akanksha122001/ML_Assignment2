import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# created model folder if not exists
os.makedirs('model', exist_ok=True)

# 1. Load Data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
print(f"Loading dataset from {url}...")
df = pd.read_csv(url)

# 2. Preprocessing
# Drop ID columns and specific failure type columns (to prevent leakage)
drop_cols = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
df = df.drop(columns=drop_cols)

# Rename columns for easier access (optional but good for code cleanliness)
df.columns = ['Type', 'Air_Temp', 'Process_Temp', 'RPM', 'Torque', 'Tool_Wear', 'Target']

# Encode 'Type' column (L, M, H)
le = LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])
joblib.dump(le, 'model/label_encoder.pkl')

X = df.drop('Target', axis=1)
y = df['Target']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale Numeric Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'model/scaler.pkl')

# Save Test Data for App Visualizations (Confusion Matrix)
joblib.dump(X_test_scaled, 'model/X_test.pkl')
joblib.dump(y_test, 'model/y_test.pkl')

# Save a CSV version of the test set (raw data) for the user to upload in the App
test_df = X_test.copy()
test_df['Target'] = y_test
test_df.to_csv('test_data.csv', index=False)
print("Saved test_data.csv for App Uploads.")

# 3. Define Models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "kNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# 4. Train and Evaluate
results = []
print("\nTraining Models...")

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    # Save model
    filename = f"model/{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
    joblib.dump(model, filename)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "MCC": round(mcc, 4)
    })

# 5. Display Results
results_df = pd.DataFrame(results)

# Save results to a CSV first
results_df.to_csv("model_evaluation_metrics.csv", index=False)
print("\nEvaluation Metrics saved to CSV.")

try:
    print(results_df.to_markdown(index=False))
except ImportError:
    print(results_df)

print("\nTraining Complete. Models and metrics saved.")
