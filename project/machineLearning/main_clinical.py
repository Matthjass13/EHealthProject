from project.data_utils import Load_data
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1️⃣ Load the files
# -----------------------------

df_patients_train = Load_data("project/dataset/data_hn_clinical_train.csv")
df_ct_train = Load_data("project/dataset/data_hn_ct_train.csv")
df_pt_train = Load_data("project/dataset/data_hn_pt_train.csv")

# -----------------------------
# 2️⃣ Clean the patients file
# -----------------------------
# Remove duplicates

print(df_patients_train.duplicated().sum())
print(df_ct_train.duplicated().sum())
print(df_pt_train.duplicated().sum())

df_patients_train = df_patients_train.drop_duplicates()
df_ct_train = df_ct_train.drop_duplicates()
df_pt_train = df_pt_train.drop_duplicates()

# Identify numerical and categorical columns
num_cols = df_patients_train.select_dtypes(include="number").columns.tolist()
cat_cols = df_patients_train.select_dtypes(include="object").columns.tolist()

# Exclude the target column (Outcome) and PatientID from imputation/encoding
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
# 3️⃣ Clean the CT and PT files
# -----------------------------
# Keep only patients present in df_patients
valid_patients = df_patients_train["PatientID"]
df_ct_train = df_ct_train[df_ct_train["PatientID"].isin(valid_patients)]
df_pt_train = df_pt_train[df_pt_train["PatientID"].isin(valid_patients)]

# -----------------------------
# 5️⃣ Check for outliers in numeric columns
# -----------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Identify numeric columns (after cleaning)
num_cols = df_patients_train.select_dtypes(include="number").columns.tolist()
if "Outcome" in num_cols:
    num_cols.remove("Outcome")
if "Performance status" in num_cols:
    num_cols.remove("Performance status")  # remove this column

# Filter non-binary columns (at least 3 unique values)
non_binary_cols = [col for col in num_cols if df_patients_train[col].nunique() > 2]

print("\n📊 Boxplots with all points for non-binary numeric columns:")
for col in non_binary_cols:
    plt.figure(figsize=(8, 4))
    # Boxplot
    sns.boxplot(x=df_patients_train[col], color='lightgray')
    # All points
    sns.stripplot(x=df_patients_train[col], color='blue', size=5, jitter=True)
    plt.title(f"Boxplot with all points: {col}")
    plt.show()

print("\n🔍 Number of outliers per non-binary column (using 1.5*IQR rule):")
for col in non_binary_cols:
    Q1 = df_patients_train[col].quantile(0.25)
    Q3 = df_patients_train[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df_patients_train[
        (df_patients_train[col] < Q1 - 1.5 * IQR)
        | (df_patients_train[col] > Q3 + 1.5 * IQR)
    ]
    print(f"{col}: {len(outliers)} outliers")

"""
# -----------------------
# CLEANING

# Duplicated lines
print("Number of duplicated rows: ",df_clinial_train.duplicated().sum())

# Show duplicated lines
print(df_clinial_train[df_clinial_train.duplicated()])

# Drop duplicated lines
df_clinial_train.drop_duplicates(inplace=True)

# --------------------------------------------------
# MISSSING VALUES

# 2️⃣ Vérifier s’il y a des valeurs manquantes
print("\n--- Vérification des valeurs manquantes ---")
missing_count = df_clinial_train.isnull().sum()
missing_percent = (missing_count / len(df_clinial_train)) * 100

# Combiner compte et pourcentage dans un seul tableau
missing_table = pd.DataFrame({
    'Missing Values': missing_count,
    'Percentage (%)': missing_percent
}).sort_values(by='Missing Values', ascending=False)

print(missing_table)

# Afficher uniquement les lignes contenant des valeurs manquantes
print("\n--- Lignes avec valeurs manquantes ---")
print(df_clinial_train[df_clinial_train.isnull().any(axis=1)])

# 🧹 Exemple : ici on choisit de remplacer les valeurs manquantes
# par la moyenne pour les colonnes numériques
df_clinial_train.fillna(df_clinial_train.mean(numeric_only=True), inplace=True)

# Vérifier après traitement
print("\n--- Vérification après traitement ---")
print(df_clinial_train.isnull().sum())"""