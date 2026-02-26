import streamlit as st
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Set page config
st.set_page_config(page_title="Beer Game AI Evaluation", layout="wide")

# Sidebar for API key
st.sidebar.title("Configuration")
google_api_key = st.sidebar.text_input("Google API Key", type="password")
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key

# --- REFINED REWARD FUNCTION LOGIC ---
# This function penalizes local costs AND ordering instability (bullwhip)
def calculate_refined_reward(game, p, current_order, prev_order, alpha=0.5, beta=0.3, gamma=0.2):
    # 1. Local Cost: Holding + Backlog
    local_cost = (game.holding_cost * game.inventory[p]) + (game.backlog_cost * game.backlog[p])
   
    # 2. Stability Penalty: Penalize high variance in ordering to tame bullwhip
    order_variance = (current_order - prev_order)**2
   
    # 3. Global Coordination: Share of the total system cost
    global_cost_share = game.total_cost / 4
   
    # Total Reward (Negative because DQN seeks to maximize)
    return -(alpha * local_cost + beta * order_variance + gamma * global_cost_share)

# --- BEHAVIORAL PERSONA DESIGN ---
# Implementation of personas: Panic-Driven Manager logic, etc.
def get_persona_order(persona_type, state, inc_order):
    inventory, backlog, pipeline = state
   
    if persona_type == 'Panic-Driven':
        # Overreacts to backlogs by ordering 2x the demand plus backlog
        return int(inc_order + (2 * backlog))
   
    elif persona_type == 'Rational-Base-Stock':
        # Attempts to maintain a steady inventory level
        target_inventory = 12
        order = inc_order + (target_inventory - inventory) + backlog
        return max(0, int(order))
   
    elif persona_type == 'Inflexible':
        # Sticks to a fixed rule regardless of system state
        return 4
       
    return inc_order # Default/Careless

# Define the BeerGame class
class BeerGame:
    def __init__(self, lead_time=2, holding_cost=0.5, backlog_cost=1.0, max_weeks=52, initial_inventory=12):
        self.lead_time = lead_time
        self.holding_cost = holding_cost
        self.backlog_cost = backlog_cost
        self.max_weeks = max_weeks
        self.initial_inventory = initial_inventory
        self.positions = ['Retailer', 'Wholesaler', 'Distributor', 'Factory']
        self.demand_pattern = [4] * 4 + [8] * (max_weeks - 4)  # Standard step change
        self.reset()

    def reset(self):
        self.week = 0
        self.inventory = [self.initial_inventory] * 4
        self.backlog = [0] * 4
        self.costs = [0] * 4
        self.total_cost = 0
        self.shipment_pipeline = [[4] * self.lead_time for _ in range(4)]  # Initial steady state
        self.history = {'inventory': [], 'backlog': [], 'costs': [], 'orders': [], 'demand': []}
        self.prev_orders = [4] * 4

    def step(self, orders):
        if self.week >= self.max_weeks:
            return True  # Done

        # Receive shipments for all positions
        for p in range(4):
            incoming_shipment = self.shipment_pipeline[p].pop(0)
            self.inventory[p] += incoming_shipment
            self.shipment_pipeline[p].append(0)

        # Process from retailer to factory
        inc_order = self.demand_pattern[self.week]
        for p in range(4):
            eff_demand = inc_order + self.backlog[p]
            shipped = min(eff_demand, self.inventory[p])
            self.inventory[p] -= shipped
            self.backlog[p] = eff_demand - shipped
            self.costs[p] += self.holding_cost * self.inventory[p] + self.backlog_cost * self.backlog[p]
            if p > 0:
                self.shipment_pipeline[p-1][-1] = shipped
            inc_order = orders[p]

        # For factory production
        self.shipment_pipeline[3][-1] = orders[3]

        self.total_cost = sum(self.costs)

        # Record history
        self.history['inventory'].append(self.inventory.copy())
        self.history['backlog'].append(self.backlog.copy())
        self.history['costs'].append(self.costs.copy())
        self.history['orders'].append(orders.copy())
        self.history['demand'].append(self.demand_pattern[self.week])

        self.prev_orders = orders.copy()
        self.week += 1
        return self.week >= self.max_weeks

    def get_state(self, p):
        pipeline_sum = sum(self.shipment_pipeline[p])
        return (self.inventory[p], self.backlog[p], pipeline_sum)

    def get_total_cost(self):
        return self.total_cost

# DQN Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size=3, action_size=21, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, lr=0.001, buffer_size=10000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size  # Orders from 0 to 20
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return np.argmax(q_values.numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states).gather(1, actions).squeeze()
        next_q_values = self.model(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# LLM Agent setup
def get_llm_chain():
    if "GOOGLE_API_KEY" not in os.environ:
        return None
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are the {role} in the Beer Game supply chain. Decide how much to order from upstream to minimize total supply chain costs. Holding cost: 0.5 per unit, Backlog cost: 1 per unit. Respond with only the integer order amount."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Current week: {week}\nInventory: {inventory}\nBacklog: {backlog}\nPipeline sum: {pipeline_sum}\nIncoming order: {inc_order}"),
    ])
    return chat_template | llm

# Main app
st.title("AI Agent Performance Evaluation in the Beer Game")

# Parameters
col1, col2 = st.columns(2)
with col1:
    lead_time = st.slider("Lead Time", 1, 5, 2)
    holding_cost = st.number_input("Holding Cost", 0.1, 2.0, 0.5)
    backlog_cost = st.number_input("Backlog Cost", 0.5, 5.0, 1.0)
with col2:
    max_weeks = st.slider("Max Weeks", 20, 100, 52)
    initial_inventory = st.number_input("Initial Inventory", 4, 20, 12)
    scenario = st.selectbox("Scenario", ["Baseline (All Rational)", "AI Agent vs Panic Managers", "All AI (LLM)", "Human-AI Collaboration", "Train DQN and Run"])

# Player types
player_types = ['Rational-Base-Stock'] * 4
if scenario == "AI Agent vs Panic Managers":
    ai_position = st.selectbox("AI Position", ["Retailer", "Wholesaler", "Distributor", "Factory"])
    p_idx = player_types.index(ai_position) if ai_position in player_types else 0  # Fix
    player_types = ['Panic-Driven'] * 4
    player_types[p_idx] = 'DQN'  # Use DQN as AI
elif scenario == "All AI (LLM)":
    player_types = ['LLM'] * 4
elif scenario == "Human-AI Collaboration":
    human_position = st.selectbox("Human Position", ["Retailer", "Wholesaler", "Distributor", "Factory"])
    human_idx = player_types.index(human_position) if human_position in player_types else 0
    player_types = ['LLM'] * 4
    player_types[human_idx] = 'Human'
elif scenario == "Train DQN and Run":
    player_types = ['DQN'] * 4

# Initialize game
if 'game' not in st.session_state:
    st.session_state.game = BeerGame(lead_time, holding_cost, backlog_cost, max_weeks, initial_inventory)

game = st.session_state.game

llm_chain = get_llm_chain()
if scenario != "Baseline (All Rational)" and llm_chain is None and 'LLM' in player_types:
    st.warning("Please enter Google API Key for LLM agents.")
    st.stop()

# For DQN
if 'dqn_agents' not in st.session_state:
    st.session_state.dqn_agents = [DQNAgent() for _ in range(4)]

dqn_agents = st.session_state.dqn_agents

# For LLM histories
if 'llm_histories' not in st.session_state:
    st.session_state.llm_histories = {pos: [] for pos in game.positions}

# Train DQN if selected
if scenario == "Train DQN and Run":
    episodes = st.number_input("Training Episodes", 100, 5000, 1000)
    if st.button("Train DQN"):
        with st.spinner("Training DQN..."):
            for ep in range(episodes):
                game.reset()
                st.session_state.llm_histories = {pos: [] for pos in game.positions}  # Reset histories
                done = False
                states = [game.get_state(p) for p in range(4)]
                while not done:
                    orders = []
                    for p in range(4):
                        action = dqn_agents[p].act(states[p])
                        orders.append(action)
                    prev_orders = game.prev_orders.copy()
                    done = game.step(orders)
                    next_states = [game.get_state(p) for p in range(4)]
                    for p in range(4):
                        reward = calculate_refined_reward(game, p, orders[p], prev_orders[p])
                        dqn_agents[p].remember(states[p], orders[p], reward, next_states[p], done)
                        dqn_agents[p].replay()
                        dqn_agents[p].decay_epsilon()
                    states = next_states
                if ep % 100 == 0:
                    st.write(f"Episode {ep}: Total Cost {game.total_cost}")
        st.success("Training complete!")

# Run simulation
if st.button("Run Simulation"):
    game.reset()
    st.session_state.llm_histories = {pos: [] for pos in game.positions}  # Reset histories
    done = False
    progress = st.progress(0)
    while not done:
        inc_order = game.demand_pattern[game.week]
        orders = [4] * 4  # Default
        states = [game.get_state(p) for p in range(4)]
        for p in range(4):
            inc_order_p = game.demand_pattern[game.week] if p == 0 else orders[p-1]
            if player_types[p] in ['Panic-Driven', 'Rational-Base-Stock', 'Inflexible']:
                orders[p] = get_persona_order(player_types[p], states[p], inc_order_p)
            elif player_types[p] == 'LLM':
                state = states[p]
                role = game.positions[p]
                history = st.session_state.llm_histories[role]
                input_dict = {
                    "role": role,
                    "week": game.week,
                    "inventory": state[0],
                    "backlog": state[1],
                    "pipeline_sum": state[2],
                    "inc_order": inc_order_p,
                    "history": history
                }
                result = llm_chain.invoke(input_dict)
                try:
                    orders[p] = int(result.content.strip())
                except:
                    orders[p] = 4  # Default if parse fail
                # Append to history
                human_content = f"Current week: {game.week}\nInventory: {state[0]}\nBacklog: {state[1]}\nPipeline sum: {state[2]}\nIncoming order: {inc_order_p}"
                history.append(HumanMessage(content=human_content))
                history.append(AIMessage(content=result.content))
            elif player_types[p] == 'DQN':
                orders[p] = dqn_agents[p].act(states[p])
            elif player_types[p] == 'Human':
                # For full sim, assume recommendation from LLM
                state = states[p]
                role = game.positions[p]
                history = st.session_state.llm_histories[role]
                input_dict = {
                    "role": role,
                    "week": game.week,
                    "inventory": state[0],
                    "backlog": state[1],
                    "pipeline_sum": state[2],
                    "inc_order": inc_order_p,
                    "history": history
                }
                result = llm_chain.invoke(input_dict)
                orders[p] = st.number_input(f"{game.positions[p]} Order (Recommendation: {result.content})", 0, 100, 4)
                # Still append to history for consistency
                human_content = f"Current week: {game.week}\nInventory: {state[0]}\nBacklog: {state[1]}\nPipeline sum: {state[2]}\nIncoming order: {inc_order_p}"
                history.append(HumanMessage(content=human_content))
                history.append(AIMessage(content=result.content))

        prev_orders = game.prev_orders.copy()
        done = game.step(orders)
        # Train DQN if present
        if 'DQN' in player_types:
            p_idx = [i for i, x in enumerate(player_types) if x == 'DQN'][0]
            reward = calculate_refined_reward(game, p_idx, orders[p_idx], prev_orders[p_idx])
            dqn_agents[p_idx].remember(states[p_idx], orders[p_idx], reward, game.get_state(p_idx), done)
            dqn_agents[p_idx].replay()
            dqn_agents[p_idx].decay_epsilon()
        progress.progress(game.week / game.max_weeks)

    st.success(f"Simulation complete. Total Cost: {game.total_cost}")

    # Display results
    st.subheader("Results")
    df = {
        "Week": list(range(1, game.week + 1)),
        "Demand": game.history['demand'],
    }
    for p in range(4):
        df[f"{game.positions[p]} Inventory"] = [h[p] for h in game.history['inventory']]
        df[f"{game.positions[p]} Backlog"] = [h[p] for h in game.history['backlog']]
        df[f"{game.positions[p]} Cost"] = [h[p] for h in game.history['costs']]
        df[f"{game.positions[p]} Order"] = [h[p] for h in game.history['orders']]

    st.dataframe(df, use_container_width=True)

    # Plots
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 1, figsize=(10, 20), sharex=True)
    for p in range(4):
        ax[p].plot(df["Week"], df[f"{game.positions[p]} Inventory"], label="Inventory")
        ax[p].plot(df["Week"], df[f"{game.positions[p]} Backlog"], label="Backlog")
        ax[p].plot(df["Week"], df[f"{game.positions[p]} Order"], label="Order")
        ax[p].set_title(game.positions[p])
        ax[p].legend()
    st.pyplot(fig)

    st.line_chart(np.array(game.history['orders']).T.tolist())
    st.caption("Order variance across stages (Visualizing the Bullwhip Effect)")

# Interactive mode for human play
st.subheader("Interactive Play (Human-AI Collaboration)")
if scenario == "Human-AI Collaboration" and st.button("Start Interactive Game"):
    game.reset()
    st.session_state.llm_histories = {pos: [] for pos in game.positions}  # Reset histories
    st.session_state.interactive = True

if 'interactive' in st.session_state and st.session_state.interactive:
    if game.week >= game.max_weeks:
        st.session_state.interactive = False
        st.success("Game Over. Total Cost: {}".format(game.total_cost))
    else:
        orders = [0] * 4
        inc_order = game.demand_pattern[game.week]
        states = [game.get_state(p) for p in range(4)]
        for p in range(4):
            state = states[p]
            role = game.positions[p]
            history = st.session_state.llm_histories[role]
            input_dict = {
                "role": role,
                "week": game.week,
                "inventory": state[0],
                "backlog": state[1],
                "pipeline_sum": state[2],
                "inc_order": inc_order,
                "history": history
            }
            if player_types[p] == 'Human':
                rec = ""
                if llm_chain:
                    result = llm_chain.invoke(input_dict)
                    rec = f" (AI Recommendation: {result.content})"
                    # Append to history
                    human_content = f"Current week: {game.week}\nInventory: {state[0]}\nBacklog: {state[1]}\nPipeline sum: {state[2]}\nIncoming order: {inc_order}"
                    history.append(HumanMessage(content=human_content))
                    history.append(AIMessage(content=result.content))
                orders[p] = st.number_input(f"{role} Order{rec}", 0, 100, 4, key=f"order_{p}_{game.week}")
            elif player_types[p] == 'LLM':
                result = llm_chain.invoke(input_dict)
                orders[p] = int(result.content.strip()) if result.content.isdigit() else 4
                # Append to history
                human_content = f"Current week: {game.week}\nInventory: {state[0]}\nBacklog: {state[1]}\nPipeline sum: {state[2]}\nIncoming order: {inc_order}"
                history.append(HumanMessage(content=human_content))
                history.append(AIMessage(content=result.content))
            inc_order = orders[p]  # Next inc_order

        if st.button("Submit Orders"):
            prev_orders = game.prev_orders.copy()
            game.step(orders)
            # Train DQN if present (though unlikely in human collab)
            if 'DQN' in player_types:
                p_idx = [i for i, x in enumerate(player_types) if x == 'DQN'][0]
                reward = calculate_refined_reward(game, p_idx, orders[p_idx], prev_orders[p_idx])
                dqn_agents[p_idx].remember(states[p_idx], orders[p_idx], reward, game.get_state(p_idx), False)
                dqn_agents[p_idx].replay()
                dqn_agents[p_idx].decay_epsilon()
            st.write(f"Week {game.week} complete. Current Total Cost: {game.total_cost}")
            st.rerun()

# Useful links
st.subheader("Useful Links and References")
links = [
    "https://infotheorylab.github.io/beer-game/",
    "https://repository.nusystem.org/bitstreams/fbfb603f-5799-427d-81cb-aea23a7c9d62/download",
    "https://docs.iza.org/dp13005.pdf",
    "https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID3992735_code2119956.pdf?abstractid=3848040&mirid=1",
    "https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID3379894_code999133.pdf?abstractid=3379894&mirid=1",
    "https://journals.bilpubgroup.com/index.php/jpr/article/download/736/844",
    "https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/j.1745-493x.2012.03285.x",
    "https://www.nature.com/articles/s41598-025-99022-8.pdf"
]
for link in links:
    st.markdown(f"- [{link}]({link})")