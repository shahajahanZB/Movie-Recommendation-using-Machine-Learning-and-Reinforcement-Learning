import pandas as pd
from joblib import load
from rl_model import RecommenderRL

# Load ML model
ml_model = load("models/ml_model.pkl")

# Assume 5 items for simplicity
num_items = 5
agent = RecommenderRL(num_items=num_items, epsilon=0.1)

# Simulate recommendations for a user
user_id = 1  # Encode user_id as in training

for _ in range(10):
    item_id = agent.recommend(user_id, ml_model)
    
    # Predict interaction using ML model
    prediction = ml_model.predict([[user_id, item_id]])[0]

    # Simulate user click/response as reward
    reward = prediction  # Use ML prediction as proxy for actual reward
    print(f"Recommended Item {item_id} => Reward: {reward}")

    agent.update(item_id, reward)
