import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
"""
Adjusted dqn_agent to handle larger batch sizes
"""
class DQNAgent:
    def __init__(self, state_size, action_size, device=None):
        self.state_size = state_size
        self.action_size = action_size

        # Replay buffer
        self.memory = deque(maxlen=20000)  # increased memory for large batch

        # Hyperparameters
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Learning rate scaled for larger batch
        self.learning_rate = 0.014

        # Device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Neural networks
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.update_target_model()

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _build_model(self):
        # Optional: larger network for better capacity
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        self.memory.append((state_tensor, action, reward, next_state_tensor, done))

    def act(self, state, strategy="epsilon", temperature=1.0):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze()

        if strategy == "epsilon":
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            return int(torch.argmax(q_values).item())

        elif strategy == "softmax":
            probs = torch.softmax(q_values / temperature, dim=0).cpu().numpy()
            return np.random.choice(len(probs), p=probs)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        states = torch.stack([s for s, _, _, _, _ in minibatch]).to(self.device)
        next_states = torch.stack([ns for _, _, _, ns, _ in minibatch]).to(self.device)
        actions = torch.LongTensor([a for _, a, _, _, _ in minibatch]).to(self.device)
        rewards = torch.FloatTensor([r for _, _, r, _, _ in minibatch]).to(self.device)
        dones = torch.FloatTensor([float(d) for _, _, _, _, d in minibatch]).to(self.device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = torch.argmax(self.model(next_states), dim=1)
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # gradient clipping
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
