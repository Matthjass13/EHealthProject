import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Charger ton dataset radiomique (par ex.)
trainSetCT = pd.read_csv("data_hn_ct_train.csv")
testSetCT = pd.read_csv("data_hn_ct_test.csv")
trainSetPT = pd.read_csv("data_hn_pt_train.csv")
testSetPT = pd.read_csv("data_hn_pt_test.csv")


def reduceDimensionality(dataset):
    # Supprimer les colonnes non numériques ou d'identifiants
    X = dataset.select_dtypes(include=[float, int])  # garde seulement les features numériques

    # ⚖️ Normaliser les données (étape essentielle avant PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 🎯 Appliquer la PCA
    pca = PCA(n_components=2)  # on garde 2 dimensions pour visualiser
    X_pca = pca.fit_transform(X_scaled)

    # 📊 Visualisation
    plt.figure(figsize=(7,5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA projection of radiomic features")
    plt.show()

    # 📈 Variance expliquée
    print("Variance explained by each component:", pca.explained_variance_ratio_)
    print("Total variance explained:", pca.explained_variance_ratio_.sum())



reduceDimensionality(trainSetCT)