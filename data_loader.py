import pandas as pd

def load_data(path='Hotel Reservations.csv'):
    return pd.read_csv(path)
