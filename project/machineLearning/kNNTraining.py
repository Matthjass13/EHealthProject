from data_utils import Load_data
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    classification_report, confusion_matrix, precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import numpy as np
import joblib
import random

# -----------------------------
# 1️⃣ Load the files
# -----------------------------
df_patients_train = Load_data("project/dataset/data_hn_clinical_train.csv")
df_ct_train = Load_data("project/dataset/data_hn_ct_train.csv")
df_pt_train = Load_data("project/dataset/data_hn_pt_train.csv")

# -----------------------------
# 2️⃣ Clean the data
# -----------------------------
print(df_patients_train.duplicated().sum())
print(df_ct_train.duplicated().sum())
print(df_pt_train.duplicated().sum())

df_patients_train = df_patients_train.drop_duplicates()
df_ct_train = df_ct_train.drop_duplicates()
df_pt_train = df_pt_train.drop_duplicates()

# Missing data report
def report_missing_values(df, name):
    print(f"\n📊 Missing values report for {name}:")
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_count,
        'Percentage (%)': missing_percent
    })
    missing_df = missing_df[missing_df['Missing Values'] > 0]
    if missing_df.empty:
        print("✅ No missing values detected.")
    else:
        print(missing_df.sort_values(by='Percentage (%)', ascending=False))

report_missing_values(df_patients_train, "Patients Train")
report_missing_values(df_ct_train, "CT Train")
report_missing_values(df_pt_train, "PT Train")

# Identify numerical and categorical columns
num_cols = df_patients_train.select_dtypes(include="number").columns.tolist()
cat_cols = df_patients_train.select_dtypes(include="object").columns.tolist()

target_col = "Outcome"
for col in [target_col, "PatientID"]:
    if col in num_cols:
        num_cols.remove(col)
    if col in cat_cols:
        cat_cols.remove(col)

# Impute missing values
imputer_num = SimpleImputer(strategy="median")
df_patients_train[num_cols] = imputer_num.fit_transform(df_patients_train[num_cols])

imputer_cat = SimpleImputer(strategy="most_frequent")
df_patients_train[cat_cols] = imputer_cat.fit_transform(df_patients_train[cat_cols])

# One-hot encode categorical columns
df_patients_train = pd.get_dummies(df_patients_train, columns=cat_cols, drop_first=True)

# -----------------------------
# 3️⃣ Prepare data for model
# -----------------------------
X = df_patients_train.drop(['Outcome', 'PatientID'], axis=1).values
y = df_patients_train['Outcome'].astype(int).values

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
joblib.dump(scaler, "scaler_knn_clinical.joblib")

# -----------------------------
# 4️⃣ Train the KNN model
# -----------------------------
knn = KNeighborsClassifier(
    n_neighbors=7,      # tu peux ajuster ce paramètre
    weights='distance', # ou 'uniform'
    metric='minkowski', # distance euclidienne
    n_jobs=-1
)
knn.fit(X_train, y_train)

# -----------------------------
# 5️⃣ Evaluation
# -----------------------------
y_proba = knn.predict_proba(X_val)[:, 1]
y_pred = (y_proba > 0.5).astype(int)

# Metrics
print("\n📈 Final Evaluation Metrics (validation set):")
print(f"ROC AUC: {roc_auc_score(y_val, y_proba):.4f}")
precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

# ROC curve
fpr, tpr, _ = roc_curve(y_val, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - kNN Clinical Model")
plt.legend()
plt.show()

# Precision-Recall curve
prec, rec, _ = precision_recall_curve(y_val, y_proba)
pr_auc = auc(rec, prec)
plt.figure()
plt.plot(rec, prec, label=f"PR AUC = {pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - kNN Clinical Model")
plt.legend()
plt.show()

# Save model
joblib.dump(knn, "best_knn_clinical_model.joblib")
print("\n💾 kNN model saved to best_knn_clinical_model.joblib")
