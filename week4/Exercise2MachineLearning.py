import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train_df = pd.read_csv("train_health_data.csv")
test_df  = pd.read_csv("test_health_data.csv")   


print("\n--- Aperçu du jeu d'entraînement ---")
print(train_df.head())




# 2️⃣ Définir la colonne cible
target_col = "Class"

# 3️⃣ Séparer features et cible
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

# Encodage de la cible (Healthy / Disease / At Risk → numérique)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Encodage des variables catégorielles (features non numériques)
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Assurer que les colonnes du train et du test correspondent
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Normaliser les données (important pour kNN et MLP)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =========================================================
# 🔹 Modèle 1 : k-Nearest Neighbors
# =========================================================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("\n--- Résultats kNN ---")
print("Accuracy :", accuracy_score(y_test, y_pred_knn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))


# =========================================================
# 🔹 Modèle 2 : Multi-Layer Perceptron (MLP)
# =========================================================
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)

print("\n--- Résultats MLP ---")
print("Accuracy :", accuracy_score(y_test, y_pred_mlp))
print("\nClassification Report:\n", classification_report(y_test, y_pred_mlp, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_mlp))

