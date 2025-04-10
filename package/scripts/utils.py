import pandas as pd

def get_accuracies(accuracies_path):
    return pd.read_csv(accuracies_path, header=0, index_col=0)
