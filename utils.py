import pandas as pd

def load_data():
    return pd.read_csv("data/large_interactions.csv")

def get_unique_users_items(data):
    return sorted(data['user_id'].unique()), sorted(data['item_id'].unique())
