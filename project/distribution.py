import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# This file display distribution bar diagrams for all variables
# Close a diagram will open the next one

trainingSet = pd.read_csv("project/dataset/data_hn_clinical_train.csv")
testSet = pd.read_csv("project/dataset/data_hn_clinical_test.csv")

def displayBarPlotDistribution(column, set, title):
    plt.figure(figsize=(6, 4))

    # Remplacer les valeurs manquantes par "Not defined" (ou autre label)
    data = set.copy()
    data[column] = data[column].fillna("Not defined")

    ax = sns.countplot(
        data=data,
        x=column,
        color="skyblue",
        edgecolor="black",
        order=sorted(data[column].unique(), key=lambda x: str(x))
    )

    # ✅ Ajout des quantités au-dessus des barres
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{int(height)}",                # texte affiché
            (p.get_x() + p.get_width() / 2., height),  # position au-dessus de la barre
            ha='center', va='bottom',
            fontsize=9, color='black', fontweight='bold'
        )
   
    plt.title(f"Distribution by {column} in the {title} set")
    plt.xlabel(column)
    plt.ylabel("Number of patients")
    plt.tight_layout()
    plt.show()

def displayPieDistribution(column, dataset, title):
    plt.figure(figsize=(6, 6))

    # Compter les occurrences de chaque catégorie
    counts = dataset[column].value_counts(dropna=False)
    labels = counts.index.astype(str)
    sizes = counts.values

    # ✅ Création du camembert
    wedges, texts, autotexts = plt.pie(
        sizes,
        labels=labels,
        autopct=lambda p: f"{p:.1f}% ({int(p * sum(sizes) / 100)})",
        startangle=90,
        colors=sns.color_palette("pastel"),
        textprops={'color': 'black', 'fontsize': 9}
    )

    # ✅ Mise en forme
    plt.title(f"Distribution by {column} in the {title} set", fontsize=12, fontweight='bold')
    plt.axis('equal')  # Assure un cercle parfait
    plt.tight_layout()
    plt.show()

def displayDistribution(column, dataset, title, max):
    # Supprimer les lignes avec âge manquant
    df_age = dataset.dropna(subset=[column])

    # Créer l'histogramme
    plt.figure(figsize=(8, 5))
    ax = sns.histplot(
        data=df_age,
        x=column,
        bins=range(30, max, 5),
        color="skyblue",
        edgecolor="black"
    )

    plt.title(f"Distribution by {column} in the {title} set")
    plt.xlabel(column)
    plt.ylabel("Number of patients")
    plt.xticks(range(30, max, 5))
    plt.tight_layout()

    # Calcul des tranches d'âge et des effectifs
    age_bins = pd.cut(df_age[column], bins=range(30, max, 5), right=False)
    age_counts = age_bins.value_counts().sort_index()

    for p in ax.patches:
        height = p.get_height()
        if height > 0:  # éviter d'afficher les 0
            ax.annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2., height),
                ha='center',
                va='bottom',
                fontsize=9,
                color='black',
                fontweight='bold'
            )

    plt.show()

'''
columns = ["CenterID", "Gender", "Tobacco", "Alcohol", "Surgery", "Chemotherapy", "Outcome"]
for column in columns:
    displayPieDistribution(column, trainingSet, "training")
    displayPieDistribution(column, testSet, "test")
'''

'''
displayDistribution("Age", trainingSet, "training", 90)
displayDistribution("Age", testSet, "test", 90)
displayDistribution("Weight", trainingSet, "training", 135)
displayDistribution("Weight", testSet, "test", 135)
'''


columns = ["Performance status"]
for column in columns:
    displayBarPlotDistribution(column, trainingSet, "training")
    displayBarPlotDistribution(column, testSet, "test")
