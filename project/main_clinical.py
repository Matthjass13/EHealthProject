from project.data_utils import Load_data
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

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
# 4️⃣ Train and evaluate model
# -----------------------------
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, classification_report, confusion_matrix
import numpy as np
import random
import joblib
import matplotlib.pyplot as plt

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Dataset (ensure targets are floats 0/1)
class ClinicalDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        # targets as float for BCEWithLogitsLoss
        self.targets = torch.FloatTensor(targets).unsqueeze(1)  # shape (N,1)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Model: remove final sigmoid, we'll use BCEWithLogitsLoss
class ClinicalNet(nn.Module):
    def __init__(self, input_size):
        super(ClinicalNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)  # logits
        )
    def forward(self, x):
        return self.network(x)

# Prepare features/targets with stratified split
X = df_patients_train.drop(['Outcome','PatientID'], axis=1).values
y = df_patients_train['Outcome'].astype(int).values  # ensure ints 0/1

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
# save scaler for later
joblib.dump(scaler, "scaler_clinical.joblib")

# Dataloaders
train_dataset = ClinicalDataset(X_train, y_train)
val_dataset = ClinicalDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# device, model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ClinicalNet(input_size=X_train.shape[1]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))  # adjust pos_weight if class imbalance
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# training with early stopping based on val AUC
num_epochs = 100
best_val_auc = 0.0
patience = 12
counter = 0
best_state = None

for epoch in range(1, num_epochs+1):
    model.train()
    train_losses = []
    for feats, targets in train_loader:
        feats = feats.to(device)
        targets = targets.to(device)
        logits = model(feats)
        loss = criterion(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # validation: gather logits and targets
    model.eval()
    val_losses = []
    preds_proba = []
    val_targets = []
    with torch.no_grad():
        for feats, targets in val_loader:
            feats = feats.to(device)
            targets = targets.to(device)
            logits = model(feats)
            loss = criterion(logits, targets)
            val_losses.append(loss.item())
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            preds_proba.extend(probs.tolist())
            val_targets.extend(targets.cpu().numpy().ravel().tolist())

    # compute AUC
    try:
        val_auc = roc_auc_score(val_targets, preds_proba)
    except ValueError:
        val_auc = 0.0

    # scheduler step (monitor AUC)
    scheduler.step(val_auc)

    # early stopping / save best
    if val_auc > best_val_auc + 1e-4:
        best_val_auc = val_auc
        best_state = model.state_dict()
        torch.save(best_state, "best_clinical_model.pth")
        counter = 0
    else:
        counter += 1

    # print periodic info
    if epoch % 5 == 0 or epoch == 1:
        train_loss_avg = np.mean(train_losses) if train_losses else 0.0
        val_loss_avg = np.mean(val_losses) if val_losses else 0.0
        # threshold 0.5 for label metrics (we'll report more later)
        preds_labels = [1 if p>0.5 else 0 for p in preds_proba]
        acc = np.mean(np.array(preds_labels) == np.array(val_targets))
        print(f"Epoch {epoch}/{num_epochs} | Train loss {train_loss_avg:.4f} | Val loss {val_loss_avg:.4f} | Val AUC {val_auc:.4f} | Val Acc {acc:.3f}")

    if counter >= patience:
        print(f"Early stopping at epoch {epoch} (no improvement in {patience} epochs).")
        break

# Load best model for final evaluation
if best_state is not None:
    model.load_state_dict(torch.load("best_clinical_model.pth"))

# Final metrics on validation set
model.eval()
probs = []
targets_all = []
with torch.no_grad():
    for feats, targets in val_loader:
        feats = feats.to(device)
        logits = model(feats)
        p = torch.sigmoid(logits).cpu().numpy().ravel()
        probs.extend(p.tolist())
        targets_all.extend(targets.cpu().numpy().ravel().tolist())

# Thresholded predictions
preds = [1 if p>0.5 else 0 for p in probs]

# Metrics
print("\nFinal Evaluation Metrics (validation set):")
print(f"Samples: {len(targets_all)}")
print(f"ROC AUC: {roc_auc_score(targets_all, probs):.4f}")
precision, recall, f1, _ = precision_recall_fscore_support(targets_all, preds, average='binary')
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
print("\nClassification report:")
print(classification_report(targets_all, preds))
print("Confusion matrix:")
print(confusion_matrix(targets_all, preds))

# ROC and PR curves
fpr, tpr, _ = roc_curve(targets_all, probs)
roc_auc = auc(fpr, tpr)
prec, rec, _ = precision_recall_curve(targets_all, probs)
pr_auc = auc(rec, prec)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve")
plt.legend()
plt.show()

plt.figure()
plt.plot(rec, prec, label=f"PR AUC = {pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall curve")
plt.legend()
plt.show()

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