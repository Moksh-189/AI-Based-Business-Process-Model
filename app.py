import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
from stable_baselines3 import PPO
from custom_env import JiraOptimizationEnv

# Page Config
st.set_page_config(page_title="AI Process Intelligence", layout="wide")

# Title and Header
st.title("🤖 AI-Powered Business Process Intelligence")
st.markdown("""
### The "Operational Blindness" Simulator
This platform compares a standard **Random/FIFO Strategy** against your **Reinforcement Learning (RL) Agent**.
Watch how the AI prioritizes High-Value tickets ($50k+) while the Random strategy gets stuck on low-value maintenance.
""")

# Sidebar for Controls
st.sidebar.header("Simulation Parameters")
num_tickets = st.sidebar.slider("Duration (Tickets)", min_value=100, max_value=2000, value=800)
speed = st.sidebar.slider("Simulation Speed", 0.001, 0.1, 0.02)
st.sidebar.markdown("---")
st.sidebar.info("Backend: Stable Baselines3 (PPO)\nActivation: GELU")

# Load Data
@st.cache_data
def load_data():
    try:
        # We use the same training data for the demo
        return pd.read_csv('training_data.csv')
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("❌ 'training_data.csv' not found. Please run your data prep script.")
    st.stop()

# Load Model
@st.cache_resource
def load_model():
    try:
        # Loading your specific GELU model
        return PPO.load("ppo_jira_agent_gelu.zip")
    except Exception as e:
        return None

model = load_model()

# The "Run" Button
if st.button("🚀 Launch Live Showdown", type="primary"):
    if not model:
        st.error("❌ Model 'ppo_jira_agent_gelu.zip' not found. Did you finish training?")
        st.stop()
        
    # Create Layout
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎲 Human / Random Strategy")
        metric_random = st.empty()
    with col2:
        st.subheader("🧠 AI Agent Strategy")
        metric_ai = st.empty()
        
    chart_placeholder = st.empty()
    
    # Initialize Environments
    env_random = JiraOptimizationEnv(df)
    env_ai = JiraOptimizationEnv(df)
    
    obs_random, _ = env_random.reset()
    obs_ai, _ = env_ai.reset()
    
    random_rev = 0
    ai_rev = 0
    
    hist_random = [0]
    hist_ai = [0]
    
    # Simulation Loop
    for i in range(num_tickets):
        # --- 1. Random Agent ---
        action_rand = env_random.action_space.sample()
        _, r_rand, _, _, _ = env_random.step(action_rand)
        random_rev += (r_rand * 1000) # Un-normalize
        
        # --- 2. AI Agent ---
        action_ai, _ = model.predict(obs_ai, deterministic=True)
        obs_ai, r_ai, _, _, _ = env_ai.step(action_ai)
        ai_rev += (r_ai * 1000)
        
        # Update Dashboard every 15 steps (to keep it smooth)
        if i % 15 == 0:
            hist_random.append(random_rev)
            hist_ai.append(ai_rev)
            
            # Metrics
            metric_random.metric("Revenue Processed", f"${random_rev:,.0f}")
            metric_ai.metric("Revenue Processed", f"${ai_rev:,.0f}", 
                             delta=f"{((ai_rev - random_rev)/max(random_rev, 1))*100:.0f}% Lift")
            
            # Live Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=hist_random, mode='lines', name='Random Strategy', line=dict(color='#ff4b4b')))
            fig.add_trace(go.Scatter(y=hist_ai, mode='lines', name='AI Agent', line=dict(color='#00c853', width=3)))
            
            fig.update_layout(
                title="Cumulative Revenue Capture",
                xaxis_title="Tickets Solved",
                yaxis_title="Total Revenue ($)",
                height=400,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            time.sleep(speed)
            
    st.success(f"Simulation Complete! The AI generated ${(ai_rev - random_rev):,.0f} more value than the baseline.")