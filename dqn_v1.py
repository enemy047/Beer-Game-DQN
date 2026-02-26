import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ----------------------------
# Beer Game Environment
# ----------------------------

class BeerGame:
    def __init__(self, lead_time=2, holding_cost=0.5, backlog_cost=1.0, max_weeks=52):
        self.lead_time = lead_time
        self.holding_cost = holding_cost
        self.backlog_cost = backlog_cost
        self.max_weeks = max_weeks
        self.positions = ['Retailer', 'Wholesaler', 'Distributor', 'Factory']
        self.reset()

    def reset(self):
        self.week = 0
        self.inventory = [12] * 4
        self.backlog = [0] * 4
        self.costs = [0] * 4
        self.total_cost = 0
        self.prev_total_cost = 0

        # randomized demand (paper-style)
        base = random.choice([3, 4, 5])
        shock = random.choice([7, 8, 9, 10])
        shock_week = random.randint(3, 10)
        self.demand_pattern = [base]*shock_week + [shock]*(self.max_weeks - shock_week)

        self.shipment_pipeline = [[4]*self.lead_time for _ in range(4)]
        self.prev_orders = [4] * 4

    def get_state(self, p):
        pipeline_sum = sum(self.shipment_pipeline[p])
        return np.array([
            self.inventory[p],
            self.backlog[p],
            pipeline_sum,
            p               # role identity
        ], dtype=np.float32)

    def step(self, orders):
        if self.week >= self.max_weeks:
            return True

        # receive shipments
        for p in range(4):
            incoming = self.shipment_pipeline[p].pop(0)
            self.inventory[p] += incoming
            self.shipment_pipeline[p].append(0)

        inc_order = self.demand_pattern[self.week]

        for p in range(4):
            demand = inc_order + self.backlog[p]
            shipped = min(demand, self.inventory[p])
            self.inventory[p] -= shipped
            self.backlog[p] = demand - shipped

            self.costs[p] += (
                self.holding_cost * self.inventory[p] +
                self.backlog_cost * self.backlog[p]
            )

            if p > 0:
                self.shipment_pipeline[p-1][-1] = shipped

            inc_order = orders[p]

        self.shipment_pipeline[3][-1] = orders[3]

        self.prev_total_cost = self.total_cost
        self.total_cost = sum(self.costs)

        self.prev_orders = orders.copy()
        self.week += 1
        return self.week >= self.max_weeks


# ----------------------------
# Shared DQN Network
# ----------------------------

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Shared DQN Agent
# ----------------------------

class SharedDQNAgent:
    def __init__(self):
        self.state_size = 4
        self.action_size = 21
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        self.batch_size = 128

        self.model = DQN(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.memory = deque(maxlen=50000)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        with torch.no_grad():
            q = self.model(torch.tensor(state))
        return torch.argmax(q).item()

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s2, d = zip(*batch)

        s = torch.tensor(s)
        s2 = torch.tensor(s2)
        a = torch.tensor(a).unsqueeze(1)
        r = torch.tensor(r)
        d = torch.tensor(d, dtype=torch.float32)

        q = self.model(s).gather(1, a).squeeze()
        q_next = self.model(s2).max(1)[0]
        target = r + self.gamma * q_next * (1 - d)

        loss = nn.MSELoss()(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self):
        torch.save(self.model.state_dict(), "shared_beer_dqn.pt")

    def load(self):
        if os.path.exists("shared_beer_dqn.pt"):
            self.model.load_state_dict(torch.load("shared_beer_dqn.pt"))


# ----------------------------
# Reward (Paper-Aligned)
# ----------------------------

def reward_fn(game, p, order, prev_order):
    local_cost = (
        game.holding_cost * game.inventory[p] +
        game.backlog_cost * game.backlog[p]
    )

    bullwhip_penalty = (order - prev_order) ** 2
    delta_cost = game.total_cost - game.prev_total_cost

    return -(
        0.5 * local_cost +
        0.3 * bullwhip_penalty +
        0.2 * delta_cost
    )


# ----------------------------
# Training Loop
# ----------------------------

if __name__ == "__main__":
    env = BeerGame()
    agent = SharedDQNAgent()
    agent.load()

    EPISODES = 3000

    for ep in range(EPISODES):
        env.reset()
        done = False

        while not done:
            states = [env.get_state(p) for p in range(4)]
            orders = [agent.act(states[p]) for p in range(4)]
            prev_orders = env.prev_orders.copy()

            done = env.step(orders)

            for p in range(4):
                r = reward_fn(env, p, orders[p], prev_orders[p])
                agent.remember(
                    states[p],
                    orders[p],
                    r,
                    env.get_state(p),
                    done
                )

            agent.replay()
            agent.decay()

        if ep % 100 == 0:
            print(f"Episode {ep} | Total Cost: {env.total_cost} | ε={agent.epsilon:.3f}")
            agent.save()