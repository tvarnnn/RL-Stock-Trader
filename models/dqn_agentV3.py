import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size, device=None):
        self.state_size = state_size  # Number of input features per observation
        self.action_size = action_size  # Number of possible actions
        self.memory = deque(maxlen=3000)  # Replay buffer to store memory
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # How fast epsilon decays per episode
        self.learning_rate = 0.005  # Learning rate

        # Device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Main network
        self.model = self._build_model().to(self.device)
        # Target network for stability
        self.target_model = self._build_model().to(self.device)
        self.update_target_model()

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    # Neural network to approximate Q-values
    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 64),  # 64 neurons
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)  # Output layer for action values
        )

    # Copy weights from main model to target model
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # Store experiences in replay buffer
    def remember(self, state, action, reward, next_state, done):
        # Convert states to tensors and move to device before storing
        state_tensor = torch.FloatTensor(state).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        self.memory.append((state_tensor, action, reward, next_state_tensor, done))

    # Choose action based on epsilon-greedy policy or softmax (Boltzmann) exploration
    def act(self, state, strategy="epsilon", temperature=1.0):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze()

        if strategy == "epsilon":
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)  # random action
            return int(torch.argmax(q_values).item())  # greedy action

        elif strategy == "softmax":
            # Convert Q-values to probabilities via softmax
            probs = torch.softmax(q_values / temperature, dim=0).cpu().numpy()
            return np.random.choice(len(probs), p=probs)

    # Train on a batch of experiences
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        # Prepare batches
        states = torch.stack([s for s, _, _, _, _ in minibatch]).to(self.device)
        next_states = torch.stack([ns for _, _, _, ns, _ in minibatch]).to(self.device)
        actions = torch.LongTensor([a for _, a, _, _, _ in minibatch]).to(self.device)
        rewards = torch.FloatTensor([r for _, _, r, _, _ in minibatch]).to(self.device)
        dones = torch.FloatTensor([float(d) for _, _, _, _, d in minibatch]).to(self.device)

        # Current Q-values
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q-values from target network (Double DQN)
        with torch.no_grad():
            next_actions = torch.argmax(self.model(next_states), dim=1)
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Loss
        loss = self.criterion(current_q, target_q)

        # Train on the entire batch at once
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
