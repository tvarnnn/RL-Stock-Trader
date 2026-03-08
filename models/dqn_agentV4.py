"""
DQNAgent with:
  - Double DQN
  - Dueling network architecture  (V(s) + A(s,a) - mean(A))
  - Prioritized Experience Replay (PER) with importance-sampling correction
  - Soft target update (Polyak averaging) for smooth target tracking
  - LayerNorm for stable activations across mixed-scale features
  - Cosine annealing LR scheduler
  - Huber loss, gradient clipping, numpy-backed buffer
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


# ── Prioritized Replay Buffer ──────────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """
    Experiences with high TD error are sampled more frequently (alpha controls how
    much prioritization is used). Importance-sampling weights (beta) correct for the
    introduced bias; beta is annealed from its initial value toward 1.0 over training.
    """
    def __init__(self, maxlen, alpha=0.6):
        self.maxlen       = maxlen
        self.alpha        = alpha        # 0 = uniform, 1 = full priority
        self.buffer       = []
        self.priorities   = np.zeros(maxlen, dtype=np.float32)
        self.pos          = 0
        self.max_priority = 1.0

    def append(self, experience):
        if len(self.buffer) < self.maxlen:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.maxlen

    def sample(self, batch_size, beta=0.4):
        n          = len(self.buffer)
        priorities = self.priorities[:n] ** self.alpha
        probs      = priorities / priorities.sum()
        indices    = np.random.choice(n, batch_size, p=probs, replace=False)

        # Importance-sampling weights — correct for sampling bias
        weights = (n * probs[indices]) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        return [self.buffer[i] for i in indices], indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = float(abs(err)) + 1e-6
        self.max_priority = float(self.priorities[:len(self.buffer)].max())

    def __len__(self):
        return len(self.buffer)


# ── Dueling Network ────────────────────────────────────────────────────────────

class DuelingDQN(nn.Module):
    """
    Shared feature extractor feeds into two separate heads:
      - Value head:     V(s)     — how good is this state overall?
      - Advantage head: A(s, a)  — how much better is each action vs. average?
    Combined: Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))

    LayerNorm after each hidden layer stabilizes activations when input features
    are on very different scales (RSI in [0,1], z-scored volume, price/1000, etc.).
    """
    def __init__(self, state_size, action_size):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x):
        features  = self.shared(x)
        value     = self.value_head(features)           # (batch, 1)
        advantage = self.advantage_head(features)       # (batch, actions)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# ── Agent ──────────────────────────────────────────────────────────────────────

class DQNAgent:
    def __init__(self, state_size, action_size, device=None, total_episodes=10):
        self.state_size  = state_size
        self.action_size = action_size

        self.memory = PrioritizedReplayBuffer(maxlen=20000)

        # Hyperparameters
        self.gamma         = 0.95
        self.epsilon       = 1.0
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # Soft target update: blends tau of online weights into target each step
        # Small tau = slow, smooth tracking; large tau approaches hard copy
        self.tau = 0.005

        # PER beta anneals from 0.4 → 1.0 (corrects importance-sampling bias)
        self.per_beta           = 0.4
        self.per_beta_increment = 0.001

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model        = DuelingDQN(state_size, action_size).to(self.device)
        self.target_model = DuelingDQN(state_size, action_size).to(self.device)
        self.update_target_model(hard=True)   # full copy at init

        # reduction="none" so we can apply per-sample IS weights before reducing
        self.criterion = nn.SmoothL1Loss(reduction="none")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Cosine annealing: decays LR from learning_rate → eta_min over training
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_episodes, eta_min=1e-5
        )

    def update_target_model(self, hard=False):
        """
        Soft (Polyak) update by default: θ_target = τ·θ_online + (1-τ)·θ_target
        Prevents the sudden Q-value jumps caused by periodic hard copies.
        Pass hard=True for a full copy (used at initialization only).
        """
        if hard:
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            for target_p, online_p in zip(self.target_model.parameters(), self.model.parameters()):
                target_p.data.copy_(self.tau * online_p.data + (1 - self.tau) * target_p.data)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((
            np.asarray(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        ))

    def act(self, state, strategy="epsilon", temperature=1.0):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze()

        if strategy == "softmax":
            probs = torch.softmax(q_values / temperature, dim=0).cpu().numpy()
            return int(np.random.choice(len(probs), p=probs))

        # Default: epsilon-greedy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return int(torch.argmax(q_values).item())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch, indices, is_weights = self.memory.sample(batch_size, beta=self.per_beta)
        self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)

        states, actions, rewards, next_states, dones = zip(*minibatch)

        states      = torch.from_numpy(np.stack(states)).to(self.device)
        next_states = torch.from_numpy(np.stack(next_states)).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)
        weights     = torch.FloatTensor(is_weights).to(self.device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: online model picks the action, target model scores it
            next_actions = torch.argmax(self.model(next_states), dim=1)
            next_q       = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q     = rewards + self.gamma * next_q * (1 - dones)

        # Update PER priorities with fresh TD errors
        td_errors = (target_q - current_q).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        # IS-weighted loss: scale each sample's loss by its correction weight
        element_loss = self.criterion(current_q, target_q)
        loss = (weights * element_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Soft target update every replay step
        self.update_target_model(hard=False)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
