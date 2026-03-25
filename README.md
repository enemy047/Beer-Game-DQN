##DEPENDENCIES##
To install dependencies:
pip install -r requirements.txt

Overview

This repository contains two implementations of the Beer Game supply chain using Reinforcement Learning:

dqn_v1.py → a clean, research-focused DQN implementation
app.py → an interactive Streamlit-based simulation with AI, LLM, and human agents

Both models simulate a four-echelon supply chain:
Retailer → Wholesaler → Distributor → Factory

The objective is to study:

cost minimization
bullwhip effect
system stability under learning# Beer-Game-DQN
Research project studying multi-agent Deep Q-Networks (DQN) for cost minimization and bullwhip suppression in the Beer Game.

dqn_v1.py — Research Model

Description
This file implements a shared Deep Q-Network (DQN) trained across all four supply chain stages.

From the code :

Each stage observes:
inventory
backlog
pipeline
role index
The agent selects discrete order quantities (0–20)
Learning happens through:
experience replay
epsilon-greedy exploration
reward shaping

Key Features
Shared DQN across all 4 entities
Randomized demand patterns (shock-based)
Reward includes:
holding + backlog cost
bullwhip penalty
system cost change
Fully automated training loop

Run Training
python dqn_v1.py

app.py — Interactive Simulation
Description

This file provides a Streamlit-based interactive interface for experimenting with different agent types.

From the code :

It supports:

DQN agents
LLM agents (Gemini via LangChain)
Human interaction
Behavioral perso

Agent Types
DQN → learned policy
LLM → decision via generative AI
Human → manual input
Personas:
Panic-Driven
Rational (Base-stock)
Inflexible

Scenarios
Baseline (all rational)
AI vs Panic managers
All LLM agents
Human-AI collaboration
Train DQN and run

Run App
streamlit run app.py

Requirements
## 🔑 API Key Setup

This project uses Google Gemini (via LangChain) for LLM-based agents.

To run LLM features, you need your own Google API key.

set GOOGLE_API_KEY=your_key_here

Features
Real-time simulation
Visualization of:
inventory
backlog
orders
Bullwhip effect visualization
Interactive gameplay

Limitations
Single node per role (linear chain)
Shared policy across all stages
No real-world data integration
Training instability across seeds
Simplified state representation

Future Work
Multi-entity supply chain (N nodes per role)
True multi-agent learning (independent policies)
Real-world demand integration
Advanced reward shaping
Robustness under disruptions

Author

Veer Kukreti
Shiv Nadar Institution of Eminence
