import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
import torch

from scripts.Data_loaderV2 import MultiStockDataLoader
from envs.stock_envV6 import BatchStockEnv
from models.dqn_agentV4 import DQNAgent

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Data ──────────────────────────────────────────────────────────────────────
tickers = ["TSLA", "AAPL", "AMZN", "GOOG"]
loader = MultiStockDataLoader(tickers)
train_data, test_data = loader.get_train_test()

# ── Environments ──────────────────────────────────────────────────────────────
train_env = BatchStockEnv(list(train_data.values()))
test_env  = BatchStockEnv(list(test_data.values()))

state_size  = train_env.observation_space.shape[0]
action_list = list(itertools.product([0, 1, 2], repeat=len(tickers)))
action_size = len(action_list)

print(f"State size: {state_size} | Action size: {action_size}")

# ── Training config ───────────────────────────────────────────────────────────
episodes   = 10
batch_size = 256
warmup_min = batch_size * 4   # experiences to collect before training starts

# ── Agent ─────────────────────────────────────────────────────────────────────
agent = DQNAgent(state_size, action_size, device=device, total_episodes=episodes)

# ── Warm-up: fill buffer with random experiences ──────────────────────────────
# Gives PER diverse data from the start so early priority estimates are meaningful
print(f"Warming up replay buffer (target: {warmup_min} experiences)...")
warmup_state = train_env.reset()
warmup_done  = np.zeros(train_env.batch_size, dtype=bool)

while len(agent.memory) < warmup_min:
    rand_idx     = random.randrange(action_size)
    rand_actions = np.tile(np.array(action_list[rand_idx]), (train_env.batch_size, 1))
    warmup_next, warmup_reward, warmup_done, _ = train_env.step(rand_actions)
    for b in range(train_env.batch_size):
        agent.remember(warmup_state[b], rand_idx, warmup_reward[b], warmup_next[b], warmup_done[b])
    if warmup_done.all():
        warmup_state = train_env.reset()
        warmup_done  = np.zeros(train_env.batch_size, dtype=bool)
    else:
        warmup_state = warmup_next

print(f"Buffer ready: {len(agent.memory)} experiences\n")

# ── Training loop ─────────────────────────────────────────────────────────────
net_worth_history  = []
test_worth_history = []

for e in range(episodes):
    state        = train_env.reset()
    done         = np.zeros(train_env.batch_size, dtype=bool)
    total_reward = 0.0

    while not done.all():
        action_idx    = agent.act(state[0])
        actions_batch = np.tile(np.array(action_list[action_idx]), (train_env.batch_size, 1))

        next_state, reward, done, _ = train_env.step(actions_batch)

        for b in range(train_env.batch_size):
            agent.remember(state[b], action_idx, reward[b], next_state[b], done[b])

        state        = next_state
        total_reward += reward.mean()

        # Train every step now that buffer is pre-filled
        agent.replay(batch_size)

    # Step LR scheduler once per episode
    agent.scheduler.step()

    # ── Greedy evaluation on test set ─────────────────────────────────────────
    test_state = test_env.reset()
    test_done  = np.zeros(test_env.batch_size, dtype=bool)

    while not test_done.all():
        saved_eps   = agent.epsilon
        agent.epsilon = 0.0
        test_idx    = agent.act(test_state[0])
        agent.epsilon = saved_eps

        test_actions = np.tile(np.array(action_list[test_idx]), (test_env.batch_size, 1))
        test_state, _, test_done, _ = test_env.step(test_actions)

    net_worth_history.append(train_env.networth.mean())
    test_worth_history.append(test_env.networth.mean())

    current_lr = agent.optimizer.param_groups[0]['lr']
    print(
        f"Episode {e+1:>2} | "
        f"Train NW: ${train_env.networth.mean():>9.2f} | "
        f"Test NW:  ${test_env.networth.mean():>9.2f} | "
        f"Reward: {total_reward:>8.4f} | "
        f"Epsilon: {agent.epsilon:.3f} | "
        f"LR: {current_lr:.6f}"
    )

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(net_worth_history, label="Train Net Worth")
plt.plot(test_worth_history, label="Test Net Worth")
plt.axhline(y=10000, color="gray", linestyle="--", alpha=0.5, label="Initial Balance")
plt.xlabel("Episode")
plt.ylabel("Net Worth ($)")
plt.title("Training vs Test Net Worth")
plt.legend()
plt.tight_layout()
plt.show()

print("Training complete!")
