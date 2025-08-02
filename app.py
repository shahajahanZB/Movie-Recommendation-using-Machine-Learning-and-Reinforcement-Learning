import streamlit as st
from utils import load_data, get_unique_users_items
from models.ml_model import train_ml_model, get_ml_recommendations
from models.rl_model import RLAgent
import numpy as np
import random

st.title("🎬 Movie Recommender using ML + RL")

data = load_data()
users, items = get_unique_users_items(data)

user = st.selectbox("Choose User", users)

svd, user_item_matrix = train_ml_model(data)
ml_recs = get_ml_recommendations(user, svd, user_item_matrix)

st.subheader("🔮 ML Recommendations")
for movie, score in ml_recs:
    st.write(f"🎞️ {movie} - Score: {round(score, 2)}")

# ---------------- RL PART ----------------
st.subheader("🧠 RL-based Exploration")

state_size = 10
action_size = len(ml_recs)
agent = RLAgent(state_size, action_size)

state = np.random.rand(state_size)
action = agent.act(state)
chosen_movie, _ = ml_recs[action]

st.markdown(f"🕹️ RL Agent recommends: **{chosen_movie}**")

reward = st.slider("Rate the recommendation (0=bad, 1=good)", 0, 1, 1)
next_state = np.random.rand(state_size)
agent.learn(state, action, reward, next_state)
