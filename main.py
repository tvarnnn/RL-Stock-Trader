import matplotlib.pyplot as plt
import torch
import itertools
import numpy as np
from scripts.Data_loaderV2 import MultiStockDataLoader
from envs.stock_envV6 import BatchStockEnv
from models.dqn_agentV3 import DQNAgent

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load stock data
tickers = ["TSLA", "AAPL", "AMZN", "GOOG"]
loader = MultiStockDataLoader(tickers)
train_data, test_data = loader.get_train_test()

# Call environments
train_env = BatchStockEnv(list(train_data.values()))
test_env = BatchStockEnv(list(test_data.values()))

state_size = train_env.observation_space.shape[0]  # observation dimension
action_list = list(itertools.product([0, 1, 2], repeat=len(tickers)))
action_size = len(action_list)

# Call DQN agent
agent = DQNAgent(state_size, action_size, device=device)

# Training parameters
episodes = 1000
batch_size = 256
net_worth_history = []
test_worth_history = []

# Training loop
for e in range(episodes):
    state = train_env.reset()
    done = np.array([False] * train_env.batch_size)
    total_reward = 0.0

    while not done.all():
        # Choose action index (epsilon-greedy) using first element of batch
        action_idx = agent.act(state[0])
        actions = np.array(action_list[action_idx])
        actions_batch = np.tile(actions, (train_env.batch_size, 1))

        # Take step in environment
        next_state, reward, done, _ = train_env.step(actions_batch)
        reward *= 0.99  # transaction penalty

        # Store experiences for all batch elements
        for b in range(train_env.batch_size):
            agent.remember(state[b], action_idx, reward[b], next_state[b], done[b])

        state = next_state
        total_reward += reward.mean()

    # Train agent if enough memory
    if len(agent.memory) >= batch_size:
        agent.replay(batch_size)

    # Update target network periodically
    if e % 10 == 0:
        agent.update_target_model()

    # evaluation on test set
    test_state = test_env.reset()
    test_done = np.array([False] * test_env.batch_size)
    test_total = 0.0

    while not test_done.all():
        # Use first element of batch for action
        test_action_idx = agent.act(test_state[0])
        test_actions = np.array(action_list[test_action_idx])
        test_actions_batch = np.tile(test_actions, (test_env.batch_size, 1))
        test_state, reward, test_done, _ = test_env.step(test_actions_batch)
        test_total += reward.mean()

    # Record mean net worth
    net_worth_history.append(train_env.networth.mean())
    test_worth_history.append(test_env.networth.mean())

    # Episode summary
    print(f"Episode {e+1} finished. Train Net Worth: {train_env.networth.mean():.2f}, "
          f"Test Net Worth: {test_env.networth.mean():.2f}, Total Reward: {total_reward:.4f}, "
          f"Epsilon: {agent.epsilon:.2f}")

# Plot net worth
plt.plot(net_worth_history, label="Train Net Worth")
plt.plot(test_worth_history, label="Test Net Worth")
plt.xlabel("Episode")
plt.ylabel("Net Worth")
plt.title("Training vs Test Net Worth")
plt.legend()
plt.show()

print("Training complete!")
