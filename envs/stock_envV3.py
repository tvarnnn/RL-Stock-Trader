import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

"""
Optimized Multi-Stock Environment for RL
- Supports any number of stocks
- Truncates monetary values to 3 decimals for efficiency
- Vectorized observation and action handling
"""

class StockEnvV3(gym.Env):
    def __init__(self, dfs, initial_balance=10000, max_steps=None, fee_pct=0.001):
        super(StockEnvV3, self).__init__()

        self.dfs = dfs
        self.n_stocks = len(dfs)
        self.initial_balance = initial_balance
        self.fee_pct = fee_pct
        self.max_steps = max_steps or min(len(df) for df in dfs) - 1

        # Action space: Buy = 1 Hold = 0 Sell = 2 for each stock
        self.action_space = spaces.MultiDiscrete([3] * self.n_stocks)

        # Observation space: price, ma5, ma10, momentum for each stock + cash + shares held
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.n_stocks * 4 + 1 + self.n_stocks,), dtype=np.float32
        )

    def reset(self):
        # Random starting step (after enough history for moving averages)
        self.current_step = random.randint(10, self.max_steps - 1)
        self.balance = float(self.initial_balance)
        self.shares_held = np.zeros(self.n_stocks, dtype=int)
        self.networth = self.balance
        self.start_networth = self.balance
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for i, df in enumerate(self.dfs):
            price = float(df.iloc[self.current_step]['Close'])
            ma5 = float(df.iloc[self.current_step-4:self.current_step+1]['Close'].mean())
            ma10 = float(df.iloc[self.current_step-9:self.current_step+1]['Close'].mean())
            momentum = price - float(df.iloc[self.current_step-1]['Close'])

            # Normalize features
            obs.extend([
                round(price / 1000, 3),
                round(ma5 / 1000, 3),
                round(ma10 / 1000, 3),
                round(momentum / 100, 3)
            ])

        # Add normalized cash balance and shares held
        obs.append(round(self.balance / (self.initial_balance * 10), 3))
        obs.extend(np.round(self.shares_held / 1000, 3))
        return np.array(obs, dtype=np.float32)

    def step(self, actions):
        """
        Executes actions for all stocks and returns new state, reward, done, info
        """
        done = False
        reward = 0

        # Execute each stock's action
        for i, action in enumerate(actions):
            price = float(self.dfs[i].iloc[self.current_step]['Close'])

            if action == 1:  # Buy
                max_shares = int(self.balance / price)
                if max_shares > 0:
                    cost = round(max_shares * price * (1 + self.fee_pct), 3)
                    self.balance = round(self.balance - cost, 3)
                    self.shares_held[i] += max_shares
                else:
                    reward -= 5  # penalty for trying to buy without cash

            elif action == 2:  # Sell
                if self.shares_held[i] > 0:
                    gain = round(self.shares_held[i] * price * (1 - self.fee_pct), 3)
                    self.balance = round(self.balance + gain, 3)
                    self.shares_held[i] = 0
                else:
                    reward -= 5  # penalty for selling without shares

        # Update net worth
        prev_networth = self.networth
        stock_values = sum(
            self.shares_held[i] * float(self.dfs[i].iloc[self.current_step]['Close'])
            for i in range(self.n_stocks)
        )
        self.networth = round(self.balance + stock_values, 3)

        # Reward = % change in net worth, clipped to [-10, 10]
        raw_reward = (self.networth - prev_networth) / prev_networth * 100 if prev_networth != 0 else 0
        reward += np.clip(raw_reward, -10, 10)

        # Step forward
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}
