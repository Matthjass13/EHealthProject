import numpy as np
import pandas as pd

# Load data

def Load_data(path):
    df = pd.read_csv(path)
    return df