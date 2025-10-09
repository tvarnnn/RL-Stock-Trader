from scripts.Data_loader import StockDataLoader
from scripts.stock_env import StockTradingEnv
from scripts.dqn_agent import DQNAgent


# Load stock data
loader = StockDataLoader("TSLA", start="2015-01-01", end="2025-01-01")
train_data, test_data = loader.get_train_test()
print("Data loaded. Sample:")
print(train_data.head())

# Initialize environment
env = StockTradingEnv(train_data)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialize agent
agent = DQNAgent(state_size, action_size)

# Training loop
episodes = 100  # Increase this for real training
batch_size = 32

for e in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done:
        step += 1
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Live step info every 10 steps or on last step
        if step % 10 == 0 or done:
            print(f"Episode {e+1}, Step {step}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Net Worth: {env.networth:.2f}")

    # Train the agent after each episode
    if len(agent.memory) >= batch_size:
        agent.replay(batch_size)

    # Episode summary
    print(f"Episode {e+1} finished. Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}, Final Net Worth: {env.networth:.2f}")

print("Training complete!")
