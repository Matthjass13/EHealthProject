import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
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
class ImagingDataset(Dataset):
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
class ImagingNet(nn.Module):
    def __init__(self, input_size):
        super(ImagingNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)

# ====================================
# 3️⃣ Préparation des données
# ====================================
def prepare_imaging_data(ct_data, pt_data, patients_data):
    # Merge CT/PT
    merged = pd.merge(ct_data, pt_data, on="PatientID", suffixes=("_ct", "_pt"))
    final = pd.merge(merged, patients_data[["PatientID", "Outcome"]], on="PatientID")

    # Supprimer les colonnes identifiantes
    for col in final.columns:
        if "PatientID" in col or "CenterID" in col:
            final = final.drop(columns=[col])

    # Encoder les colonnes catégorielles
    cat_cols = final.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        final = pd.get_dummies(final, columns=cat_cols, drop_first=True)

    # Supprimer les colonnes constantes
    final = final.loc[:, final.nunique() > 1]

    # Corrélation avec Outcome
    corr = final.corr(numeric_only=True)["Outcome"].abs().sort_values(ascending=False)
    print("\n🔍 Top 10 correlated features with Outcome:")
    print(corr.head(10))

    # Identifier les features trop corrélées à Outcome
    high_corr = corr[corr > 0.95].index.tolist()

    # Supprimer seulement si ≠ Outcome
    high_corr = [c for c in high_corr if c != "Outcome"]
    if len(high_corr) > 0:
        print(f"\n⚠️ Removing highly correlated features: {high_corr}")
        final = final.drop(columns=[col for col in high_corr if col in final.columns])

    # Vérifier que la colonne Outcome est bien là
    if "Outcome" not in final.columns:
        raise ValueError("❌ 'Outcome' is missing from merged data — check dataset columns!")

    # Séparer X et y
    X = final.drop(columns=["Outcome"]).values
    y = final["Outcome"].astype(int).values

    return X, y

    # Merge CT/PT
    merged = pd.merge(ct_data, pt_data, on="PatientID", suffixes=("_ct", "_pt"))
    final = pd.merge(merged, patients_data[["PatientID", "Outcome"]], on="PatientID")

    # Supprimer les colonnes non pertinentes
    for col in final.columns:
        if "PatientID" in col or "CenterID" in col:
            final = final.drop(columns=[col])

    # Encoder les colonnes catégorielles
    cat_cols = final.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        final = pd.get_dummies(final, columns=cat_cols, drop_first=True)

    # Supprimer les colonnes constantes
    final = final.loc[:, final.nunique() > 1]

    # Vérifier les corrélations
    corr = final.corr(numeric_only=True)["Outcome"].abs().sort_values(ascending=False)
    print("\n🔍 Top 10 correlated features with Outcome:")
    print(corr.head(10))

    # Supprimer les colonnes suspectes (corr > 0.95)
    high_corr = corr[corr > 0.95].index.tolist()
    if len(high_corr) > 1:
        print(f"\n⚠️ Removing highly correlated features: {high_corr[1:]}")
        final = final.drop(columns=[col for col in high_corr[1:] if col in final.columns])

    # Séparer features / target
    if "Outcome" not in final.columns:
        raise ValueError("❌ La colonne 'Outcome' a été supprimée trop tôt — vérifie tes données d'entrée.")

    X = final.drop(columns=["Outcome"]).values
    y = final["Outcome"].astype(int).values

    return X, y

# ====================================
# 4️⃣ Programme principal
# ====================================
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from project.data_utils import Load_data
    except ImportError:
        from data_utils import Load_data

    # Charger les fichiers
    df_ct = Load_data("project/dataset/data_hn_ct_train.csv")
    df_pt = Load_data("project/dataset/data_hn_pt_train.csv")
    df_patients = Load_data("project/dataset/data_hn_clinical_train.csv")

    # Préparation des données
    X, y = prepare_imaging_data(df_ct, df_pt, df_patients)

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
    joblib.dump(scaler, "scaler_imaging.joblib")

    # Datasets & loaders
    train_loader = DataLoader(ImagingDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(ImagingDataset(X_val, y_val), batch_size=64, shuffle=False)

    # Modèle, loss, optim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImagingNet(input_size=X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    # ====================================
    # 5️⃣ Entraînement
    # ====================================
    best_auc = 0
    patience = 10
    counter = 0
    best_state = None
    num_epochs = 100

    for epoch in range(1, num_epochs + 1):
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

        # Validation
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

        # Sauvegarde du meilleur modèle
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = model.state_dict()
            torch.save(best_state, "best_imaging_model.pth")
            counter = 0
        else:
            counter += 1

        if epoch % 5 == 0 or epoch == 1:
            acc = np.mean((np.array(preds) > 0.5) == np.array(labels))
            print(f"Epoch {epoch:03d} | Train Loss {np.mean(train_losses):.4f} | Val Loss {val_loss:.4f} | AUC {val_auc:.4f} | Acc {acc:.3f}")

        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # ====================================
    # 6️⃣ Évaluation finale
    # ====================================
    model.load_state_dict(torch.load("best_imaging_model.pth"))
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
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, preds_bin, average="binary")

    print("\n📊 Final evaluation on validation set:")
    print(f"AUC: {auc_score:.4f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
    print("\nClassification report:")
    print(classification_report(all_targets, preds_bin))
    print("Confusion matrix:")
    print(confusion_matrix(all_targets, preds_bin))

    # ====================================
    # 7️⃣ ROC & Precision-Recall curves
    # ====================================
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    prec, rec, _ = precision_recall_curve(all_targets, all_probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc_score:.3f}")
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(rec, prec, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()
