import matplotlib.pyplot as plt
from scripts.Data_loader import StockDataLoader
from envs.stock_envV2 import StockEnvV2
from models.dqn_agentV2 import DQNAgent

# Load stock data
loader = StockDataLoader("TSLA", start="2015-01-01", end="2025-01-01")
train_data, test_data = loader.get_train_test()
print("Data loaded. Sample:")
print(train_data.head())

# Initialize environments
train_env = StockEnvV2(train_data)
test_env = StockEnvV2(test_data)

state_size = train_env.observation_space.shape[0]  # 6 now in StockEnvV2
action_size = train_env.action_space.n

# Initialize agent
agent = DQNAgent(state_size, action_size)

# Training loop parameters
episodes = 100
batch_size = 32
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
        action = agent.act(state)
        next_state, reward, done, _ = train_env.step(action)

        if action != 0:  # If buying or selling
            reward *= 0.99

        # Remember experience for replay
        agent.remember(state, action, reward, next_state, done)
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
        test_action = agent.act(test_state)
        test_state, reward, test_done, _ = test_env.step(test_action)
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
