import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

"""
improved stock environment for RL
    Added:
    -Expanded state (price, moving averages, balances, shares held, momentum
    -Reward shaping
    -Transaction fees
    -Normalized inputs
    -Randomized episode start
"""
class StockEnvV2(gym.Env):

    def __init__(self, df, initial_balance = 10000, max_steps = None, fee_pct=0.001):
        super(StockEnvV2, self).__init__()
        self.df = df.reset_index(drop = True)
        self.initial_balance = initial_balance
        self.fee_pct = fee_pct
        self.max_steps = max_steps or len(df) - 1

        # actions
        self.action_space = spaces.Discrete(3)

        # Observations
        self.observation_space = spaces.Box(
            low = 0, high = 1, shape = (6,), dtype = np.float32
        )

    def reset(self):
        # Random starting point for episode
        self.current_step = random.randint(10, len(self.df) - 100)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.networth = self.balance
        self.start_networth = self.balance
        return self._get_obs()

    def _get_obs(self):
        price = float(self.df.loc[self.current_step, 'Close'])

        # Moving averages
        ma5 = float(self.df.loc[self.current_step-4:self.current_step, 'Close'].mean())
        ma10 = float(self.df.loc[self.current_step-9:self.current_step, 'Close'].mean())

        # Momentum - Price change from previous day
        momentum = price - float(self.df.loc[self.current_step-1, 'Close'])

        # Normalizing inputs
        price_norm = price / 1000 # Rough estimate
        ma5_norm = ma5 / 1000
        ma10_norm = ma10 / 1000
        momentum_norm = momentum / 100
        balance_norm = self.balance / (self.initial_balance * 10)
        shares_norm = self.shares_held / 1000

        return np.array([price_norm, ma5_norm, ma10_norm, momentum_norm, balance_norm, shares_norm], dtype=np.float32)

    def step(self, action):
        done = False
        reward = 0

        price = float(self.df.loc[self.current_step, 'Close'])

        # Execute actions with transaction fees
        if action == 1: # Buy action
            max_shares = int(self.balance / price)
            if max_shares > 0:
                cost = max_shares * price * (1 + self.fee_pct)
                self.balance -= cost
                self.shares_held += max_shares
            else:
                reward -= 5 # Penalty for trying to buy without money

        elif action == 2: # Sell action
            if self.shares_held > 0:
                gain = self.shares_held * price * (1 -  self.fee_pct)
                self.balance += gain
                self.shares_held = 0
            else:
                reward -= 5 # Penalty for selling with no shares

        # Updating net worth
        prev_networth = self.networth
        self.networth = self.balance + self.shares_held * price

        # Reward: % changes in net worth plus risk-adjusted bonuses
        raw_reward = (self.networth - prev_networth) / prev_networth * 100 if prev_networth != 0 else 0
        reward += np.clip(raw_reward, -10, 10)
        # Step forward
        self.current_step += 1
        if self.current_step >= len(self.df) - 1 or self.current_step >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}
