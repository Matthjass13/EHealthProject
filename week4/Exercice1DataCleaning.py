import pandas as pd


# 🔹 1. Lire le fichier CSV
df = pd.read_csv("iris_to_clean.csv")  



# --------------------------------------------------
# DUPLICATES

# 🔹 2. Afficher les 5 premières lignes
#print(df.head())
# 🔹 3. Vérifier les infos sur les colonnes
#print(df.info())

# 4 lignes dupliquées
print(df.duplicated().sum())

# Afficher les lignes dupliquées
print(df[df.duplicated()])

# Effacer les lignes dupliquées
df.drop_duplicates(inplace=True)
df.to_csv("iris_to_clean.csv", index=False)





# --------------------------------------------------
# MISSSING VALUES

# 2️⃣ Vérifier s’il y a des valeurs manquantes
print("\n--- Vérification des valeurs manquantes ---")
missing_count = df.isnull().sum()
missing_percent = (missing_count / len(df)) * 100

# Combiner compte et pourcentage dans un seul tableau
missing_table = pd.DataFrame({
    'Missing Values': missing_count,
    'Percentage (%)': missing_percent
}).sort_values(by='Missing Values', ascending=False)

print(missing_table)

# Afficher uniquement les lignes contenant des valeurs manquantes
print("\n--- Lignes avec valeurs manquantes ---")
print(df[df.isnull().any(axis=1)])

# 🧹 Exemple : ici on choisit de remplacer les valeurs manquantes
# par la moyenne pour les colonnes numériques
df.fillna(df.mean(numeric_only=True), inplace=True)

# Vérifier après traitement
print("\n--- Vérification après traitement ---")
print(df.isnull().sum())

# (Optionnel) Sauvegarder le dataset nettoyé
df.to_csv("iris_no_missing_values.csv", index=False)
print("\n✅ Dataset nettoyé enregistré sous 'iris_no_missing_values.csv'")




# --------------------------------------------------
# OUTLIERS

print("\n--- Vérification des outliers ---")

# 2️⃣ Détection des outliers avec la méthode IQR (Interquartile Range)
# Cette méthode est robuste et très utilisée pour identifier les valeurs extrêmes

def detect_outliers_iqr(data):
    outlier_indices = []
    
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Trouver les indices où la valeur est en dehors des bornes
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index
        outlier_indices.extend(outliers)
        
        print(f"Colonne '{col}' → bornes [{lower_bound:.2f}, {upper_bound:.2f}], outliers trouvés : {len(outliers)}")
    
    # Retourner les indices uniques
    return list(set(outlier_indices))

# 3️⃣ Identifier les outliers
outliers = detect_outliers_iqr(df)
print(f"\nNombre total de lignes contenant des outliers : {len(outliers)}")

# Afficher les lignes suspectes (si tu veux les examiner avant suppression)
print("\n--- Lignes suspectes ---")
print(df.loc[outliers])

# Supprimer les *non-true* outliers (valeurs aberrantes clairement erronées)
# Ici on suppose qu’on supprime toutes les lignes détectées comme outliers :
df_clean = df.drop(index=outliers)

# Vérifier le résultat
print(f"\n✅ Nombre de lignes avant nettoyage : {len(df)}")
print(f"✅ Nombre de lignes après suppression des outliers : {len(df_clean)}")

# (Optionnel) Sauvegarder le dataset propre
df_clean.to_csv("iris_no_outliers.csv", index=False)
print("\n✅ Dataset nettoyé enregistré sous 'iris_no_outliers.csv'")
