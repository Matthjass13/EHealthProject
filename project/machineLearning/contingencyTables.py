import pandas as pd

from tabulate import tabulate

# This file creates contingency tables for some interesting pairs of fields
# We can see the result both in absolute value and in percentage.

trainingSet = pd.read_csv("project/dataset/data_hn_clinical_train.csv")
testSet = pd.read_csv("project/dataset/data_hn_clinical_test.csv")

def displayContingencyTable(column1, column2, set):
    table = pd.crosstab(set[column1], set[column2])
    print("Contingency table (raw counts):")
    print(tabulate(table, headers='keys', tablefmt='grid'))
    print()

def displayContingencyTableInPercentage(column1, column2, set):
    table = pd.crosstab(set[column1], set[column2], normalize='index') * 100
    print("Contingency table (percentages):")
    print(tabulate(table.round(2), headers='keys', tablefmt='grid'))
    print()

def displayContingency(column1, column2, set):
    print(f"=== Tables for '{column1}' / '{column2}' ===\n")
    displayContingencyTable(column1, column2, set)
    #displayContingencyTableInPercentage(column1, column2, set)
    print("=" * 40 + "\n")

displayContingency("Performance status", "Outcome", trainingSet)
displayContingency("Performance status", "Outcome", testSet)

displayContingency("Surgery", "Outcome", trainingSet)
displayContingency("Surgery", "Outcome", testSet)

displayContingency("Chemotherapy", "Outcome", trainingSet)
displayContingency("Chemotherapy", "Outcome", testSet)

displayContingency("Tobacco", "Outcome", trainingSet)
displayContingency("Tobacco", "Outcome", testSet)

displayContingency("Alcohol", "Outcome", trainingSet)
displayContingency("Alcohol", "Outcome", testSet)


