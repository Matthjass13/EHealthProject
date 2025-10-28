import pandas as pd

# This file computes discrete statistics on all numerical variable.

trainingSet = pd.read_csv("project/dataset/data_hn_clinical_train.csv")
testSet = pd.read_csv("project/dataset/data_hn_clinical_test.csv")

def displayStats(datasetFile):
    print("Statistics :\n")

    colonnes = ["Tobacco", "Alcohol", "Performance status", "Surgery", 
                "Chemotherapy", "Age", "Weight", "Outcome"]
    noms = {
        "Tobacco": "Tobacco",
        "Alcohol": "Alcohol",
        "Performance status": "Performance status",
        "Surgery": "Surgery",
        "Chemotherapy": "Chemotherapy",
        "Age": "Age",
        "Weight": "Weight",
        "Outcome": "Outcome"
    }

    for col in colonnes:
        data = datasetFile[col].dropna()
        moyenne = round(computeMean(data), 2)
        mediane = round(computeMedian(data), 2)
        mode = computeMode(data)
        mode_affiche = round(mode, 2) if isinstance(mode, (int, float)) else mode
        etendue = round(computeRange(data), 2)
        variance = round(computeVariance(data), 2)
        std_dev = round(computeStdDev(data), 2)

        print(f"{noms[col]:<20} → Mean : {moyenne} | Mediane : {mediane} | Mode : {mode_affiche} | "
              f"Range : {etendue} | Variance : {variance} | Standard Deviation : {std_dev}")

def computeMean(data):
    return data.mean()

def computeMedian(data):
    return data.median()

def computeMode(data):
    mode_series = data.mode()
    return mode_series.iloc[0] if not mode_series.empty else "N/A"

def computeRange(data):
    return data.max() - data.min()

def computeVariance(data):
    return data.var()

def computeStdDev(data):
    return data.std()

displayStats(trainingSet)
displayStats(testSet)