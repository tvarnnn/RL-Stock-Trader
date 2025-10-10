import matplotlib.pyplot as plt
import torch
import itertools
from scripts.Data_loaderV2 import MultiStockDataLoader
from envs.stock_envV5 import StockEnvV5  # Reward system overhaul
from models.dqn_agentV3 import DQNAgent

# Use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# Load multi-stock data
tickers = ["TSLA", "AAPL", "AMZN", "GOOG"]
loader = MultiStockDataLoader(tickers)
train_data, test_data = loader.get_train_test()

# Initialize environments

# Pass a list of DataFrames to StockEnvV3
train_env = StockEnvV5(list(train_data.values()))
test_env = StockEnvV5(list(test_data.values()))

state_size = train_env.observation_space.shape[0]  # State vector size

# Flatten MultiDiscrete action space

# Generate all possible combinations of actions for N stocks
action_list = list(itertools.product([0, 1, 2], repeat=len(tickers)))
action_size = len(action_list)  # Total discrete actions = 3^N

# Initialize DQN agent
agent = DQNAgent(state_size, action_size, device=device)

# Training loop parameters
episodes = 1000
batch_size = 256
net_worth_history = []
test_worth_history = []

# Training loop
for e in range(episodes):
    state = train_env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done:
        step += 1
        # Choose action index using DQN agent (epsilon-greedy)
        action_idx = agent.act(state)
        # Map discrete index to multi-stock action vector
        actions = action_list[action_idx]

        # Take step in environment
        next_state, reward, done, _ = train_env.step(actions)

        # Penalize transaction slightly if buying or selling
        if any(a != 0 for a in actions):
            reward *= 0.99

        # Store experience in replay buffer
        agent.remember(state, action_idx, reward, next_state, done)
        state = next_state
        total_reward += reward

    # Train agent after each episode
    if len(agent.memory) >= batch_size:
        agent.replay(batch_size)

    # Update target network every 10 episodes for stability
    if e % 10 == 0:
        agent.update_target_model()

    # Evaluate on test set (no learning)
    test_state = test_env.reset()
    test_done = False
    test_total = 0
    while not test_done:
        test_action_idx = agent.act(test_state)
        test_actions = action_list[test_action_idx]
        test_state, reward, test_done, _ = test_env.step(test_actions)
        test_total += reward

    net_worth_history.append(train_env.networth)
    test_worth_history.append(test_env.networth)

    # Episode summary
    print(f"Episode {e+1} finished. Train Net Worth: {train_env.networth:.2f}, "
          f"Test Net Worth: {test_env.networth:.2f}, Total Reward: {total_reward:.2f}, "
          f"Epsilon: {agent.epsilon:.2f}")

# Plot net worth over episodes
plt.plot(net_worth_history, label="Train Net Worth")
plt.plot(test_worth_history, label="Test Net Worth")
plt.xlabel("Episode")
plt.ylabel("Net Worth")
plt.title("Training vs Test Net Worth")
plt.legend()
plt.show()

print("Training complete!")
