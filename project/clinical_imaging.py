import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    classification_report, confusion_matrix, precision_recall_fscore_support
)
import numpy as np
import pandas as pd
import joblib
import random
import matplotlib.pyplot as plt
import sys
import os

# ====================================
# 0️⃣ Reproductibilité
# ====================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ====================================
# 1️⃣ Dataset PyTorch
# ====================================
class CombinedDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets).unsqueeze(1)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# ====================================
# 2️⃣ Réseau de neurones MLP
# ====================================
class CombinedNet(nn.Module):
    def __init__(self, input_size):
        super(CombinedNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)

# ====================================
# 3️⃣ Préparation des données
# ====================================

def prepare_combined_data(clinical_data, ct_data, pt_data):
    # Fusion des données d'imagerie
    imaging_data = pd.merge(ct_data, pt_data, on="PatientID", suffixes=("_ct", "_pt"))
    
    # Fusion avec les données cliniques
    final_data = pd.merge(imaging_data, clinical_data, on="PatientID")

    # Supprimer les colonnes identifiantes
    id_cols = [col for col in final_data.columns if "ID" in col and col != "PatientID"]
    final_data = final_data.drop(columns=id_cols + ["PatientID"])

    # Gérer les valeurs manquantes d'abord
    # 1. Afficher le nombre de NaN par colonne
    nan_counts = final_data.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        print("\n📊 Missing values per column:")
        print(nan_cols)
    
    # 2. Remplacer les NaN
    # Pour les colonnes numériques
    numeric_cols = final_data.select_dtypes(include=['float64', 'int64']).columns
    final_data[numeric_cols] = final_data[numeric_cols].fillna(final_data[numeric_cols].median())
    
    # Pour les colonnes catégorielles
    cat_cols = final_data.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        final_data[cat_cols] = final_data[cat_cols].fillna('missing')
        final_data = pd.get_dummies(final_data, columns=cat_cols, drop_first=True)

    # Supprimer les colonnes constantes
    final_data = final_data.loc[:, final_data.nunique() > 1]

    # Corrélation avec Outcome
    corr = final_data.corr(numeric_only=True)["Outcome"].abs().sort_values(ascending=False)
    print("\n🔍 Top 10 correlated features with Outcome:")
    print(corr.head(10))

    # Supprimer features trop corrélées
    high_corr = corr[corr > 0.95].index.tolist()
    high_corr = [c for c in high_corr if c != "Outcome"]
    if len(high_corr) > 0:
        print(f"\n⚠️ Removing highly correlated features: {high_corr}")
        final_data = final_data.drop(columns=[col for col in high_corr if col in final_data.columns])

    # Vérifier qu'il ne reste plus de NaN
    if final_data.isna().any().any():
        raise ValueError("❌ Il reste des valeurs manquantes après le prétraitement!")

    X = final_data.drop(columns=["Outcome"]).values
    y = final_data["Outcome"].astype(int).values
    return X, y

    # Fusion des données d'imagerie
    imaging_data = pd.merge(ct_data, pt_data, on="PatientID", suffixes=("_ct", "_pt"))
    
    # Fusion avec les données cliniques
    final_data = pd.merge(imaging_data, clinical_data, on="PatientID")

    # Supprimer les colonnes identifiantes
    id_cols = [col for col in final_data.columns if "ID" in col and col != "PatientID"]
    final_data = final_data.drop(columns=id_cols + ["PatientID"])

    # Encoder les colonnes catégorielles
    cat_cols = final_data.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        final_data = pd.get_dummies(final_data, columns=cat_cols, drop_first=True)

    # Gérer les valeurs manquantes
    # 1. Afficher le nombre de NaN par colonne
    nan_counts = final_data.isna().sum()
    print("\n📊 Missing values per column:")
    print(nan_counts[nan_counts > 0])
    
    # 2. Remplacer les NaN par la médiane pour les colonnes numériques
    numeric_cols = final_data.select_dtypes(include=['float64', 'int64']).columns
    final_data[numeric_cols] = final_data[numeric_cols].fillna(final_data[numeric_cols].median())

    # Supprimer les colonnes constantes
    final_data = final_data.loc[:, final_data.nunique() > 1]

    # Fusion des données d'imagerie
    imaging_data = pd.merge(ct_data, pt_data, on="PatientID", suffixes=("_ct", "_pt"))
    
    # Fusion avec les données cliniques
    final_data = pd.merge(imaging_data, clinical_data, on="PatientID")

    # Supprimer les colonnes identifiantes
    id_cols = [col for col in final_data.columns if "ID" in col and col != "PatientID"]
    final_data = final_data.drop(columns=id_cols + ["PatientID"])

    # Encoder les colonnes catégorielles
    cat_cols = final_data.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        final_data = pd.get_dummies(final_data, columns=cat_cols, drop_first=True)

    # Supprimer les colonnes constantes
    final_data = final_data.loc[:, final_data.nunique() > 1]

    # Corrélation avec Outcome
    corr = final_data.corr(numeric_only=True)["Outcome"].abs().sort_values(ascending=False)
    print("\n🔍 Top 10 correlated features with Outcome:")
    print(corr.head(10))

    # Supprimer features trop corrélées
    high_corr = corr[corr > 0.95].index.tolist()
    high_corr = [c for c in high_corr if c != "Outcome"]
    if len(high_corr) > 0:
        print(f"\n⚠️ Removing highly correlated features: {high_corr}")
        final_data = final_data.drop(columns=[col for col in high_corr if col in final_data.columns])

    X = final_data.drop(columns=["Outcome"]).values
    y = final_data["Outcome"].astype(int).values
    return X, y

if __name__ == "__main__":
    # Add project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    
    try:
        from project.machineLearning.data_utils import Load_data
    except ImportError:
        try:
            from machineLearning.data_utils import Load_data
        except ImportError:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from machineLearning.data_utils import Load_data

    # Charger les données
    df_clinical = Load_data("project/dataset/data_hn_clinical_train.csv")
    df_ct = Load_data("project/dataset/data_hn_ct_train.csv")
    df_pt = Load_data("project/dataset/data_hn_pt_train.csv")

    # Préparation des données
    X, y = prepare_combined_data(df_clinical, df_ct, df_pt)
    print(f"\n✅ Dataset ready: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class balance: {np.bincount(y)}")

    # Split stratifié
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Standardisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    joblib.dump(scaler, "scaler_combined.joblib")

    # Réduction de dimension PCA
    print("\n⚙️ Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=0.95, random_state=SEED)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    joblib.dump(pca, "pca_combined.joblib")
    print(f"✅ PCA reduced dimensionality to {X_train.shape[1]} components")

    # Datasets & loaders
    train_loader = DataLoader(CombinedDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(CombinedDataset(X_val, y_val), batch_size=64, shuffle=False)

    # Modèle, loss, optim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombinedNet(input_size=X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # ====================================
    # 5️⃣ Entraînement
    # ====================================
    best_auc = 0
    patience = 12
    counter = 0
    best_state = None
    num_epochs = 100

    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        train_losses = []
        for feats, targets in train_loader:
            feats, targets = feats.to(device), targets.to(device)
            logits = model(feats)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation phase
        model.eval()
        val_losses, preds, labels = [], [], []
        with torch.no_grad():
            for feats, targets in val_loader:
                feats, targets = feats.to(device), targets.to(device)
                logits = model(feats)
                loss = criterion(logits, targets)
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                val_losses.append(loss.item())
                preds.extend(probs)
                labels.extend(targets.cpu().numpy().ravel())

        val_auc = roc_auc_score(labels, preds)
        val_loss = np.mean(val_losses)
        scheduler.step(val_auc)

        # Early stopping / save best
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = model.state_dict()
            torch.save(best_state, "best_combined_model.pth")
            counter = 0
        else:
            counter += 1

        if epoch % 5 == 0 or epoch == 1:
            acc = np.mean((np.array(preds) > 0.5) == np.array(labels))
            print(f"Epoch {epoch:03d} | Train Loss {np.mean(train_losses):.4f} | "
                  f"Val Loss {val_loss:.4f} | AUC {val_auc:.4f} | Acc {acc:.3f}")

        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # ====================================
    # 6️⃣ Évaluation finale
    # ====================================
    model.load_state_dict(torch.load("best_combined_model.pth"))
    model.eval()
    all_probs, all_targets = [], []
    
    with torch.no_grad():
        for feats, targets in val_loader:
            feats = feats.to(device)
            logits = model(feats)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            all_probs.extend(probs)
            all_targets.extend(targets.numpy().ravel())

    preds_bin = [1 if p > 0.5 else 0 for p in all_probs]
    auc_score = roc_auc_score(all_targets, all_probs)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, preds_bin, average="binary"
    )

    print("\n📊 Final evaluation on validation set:")
    print(f"AUC: {auc_score:.4f} | Precision: {precision:.3f} | "
          f"Recall: {recall:.3f} | F1: {f1:.3f}")
    print("\nClassification report:")
    print(classification_report(all_targets, preds_bin))
    print("\nConfusion matrix:")
    print(confusion_matrix(all_targets, preds_bin))

    # ====================================
    # 7️⃣ ROC & Precision-Recall curves
    # ====================================
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    prec, rec, _ = precision_recall_curve(all_targets, all_probs)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc_score:.3f}")
    plt.plot([0,1], [0,1], '--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    pr_auc = auc(rec, prec)
    plt.subplot(1, 2, 2)
    plt.plot(rec, prec, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    
    plt.tight_layout()
    plt.show()