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

    def step(self, action):
        price = float(self.df.loc[self.current_step, 'Close'])

        # Default penalty/reward
        reward = 0

        # Executable actions
        if action == 1:  # Buy
            max_shares = int(self.balance / price)
            if max_shares > 0:
                self.balance -= max_shares * price
                self.shares_held += max_shares
            else:
                reward -= 10  # Penalty for trying to buy without enough balance

        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.balance += self.shares_held * price
                self.shares_held = 0
            else:
                reward -= 10  # Penalty for trying to sell with no shares

        # Update net worth
        prev_networth = self.networth
        self.networth = self.balance + self.shares_held * price

        # Reward proportional to % change in net worth
        pct_change = (self.networth - prev_networth) / prev_networth if prev_networth != 0 else 0
        reward += pct_change * 100  # scale factor to keep rewards visible

        # Next step
        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self._get_obs(), reward, done, {}