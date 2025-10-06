import pandas as pd
from app import config

def load_data():
    try:
        df = pd.read_csv(config.SOURCE_CSV_PATH)
        return df
    except FileNotFoundError:
        raise Exception(f"CSV file not found at {config.SOURCE_CSV_PATH}")
