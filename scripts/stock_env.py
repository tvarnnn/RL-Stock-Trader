import gymnasium as gym
from gymnasium import spaces
import numpy as np

class StockTradingEnv(gym.Env):
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        self.df = df.reset_index()
        self.max_steps = len(df) - 1 # total days we can trade
        self.current_step = 0 # Current day in the environment
        self.balance = 10000 # Starting cash for trading
        self.shares_held = 0 # How many shares the agent has
        self.networth = self.balance

        # Actions: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3) # 3 possible actions
        # Observations: [Current price, balance, shares held]
        self.observation_space = spaces.Box(
            low = 0,
            high = np.inf,
            shape = (3,), dtype = np.float32
        )

    def reset(self): # Called at the start and resets when done
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.networth = self.balance
        return self._get_obs() # Returns observation

    def _get_obs(self):
        price = float(self.df.loc[self.current_step, 'Close'])
        return np.array([price, self.balance, self.shares_held], dtype=np.float32)

    def step(self, action): # Executes 1 action in the environment
        price = float(self.df.loc[self.current_step, 'Close']) # Current price

        # Executable actions
        if action == 1: # Buying action
            # buy as many possible shares
            max_shares = int(self.balance / price)
            self.balance -= max_shares * price
            self.shares_held += max_shares

        elif action == 2: # Selling action
            self.balance += self.shares_held * price
            self.shares_held = 0

        self.networth = self.balance +self.shares_held * price # Net worth

        if self.current_step == 0: # Reward for changes in net worth
            reward = 0
        else:
            prev_price = float(self.df.loc[self.current_step-1, 'Close'])
            prev_networth = self.balance + self.shares_held * prev_price
            reward = self.networth - prev_networth

        # Next step
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Return Observation, reward, done, info
        return self._get_obs(), reward, done, {}