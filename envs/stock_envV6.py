import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BatchStockEnv(gym.Env):
    """
    Vectorized multi-stock trading environment with precomputed prices,
    moving averages (ma5, ma10), and momentum for observations.
    Rewards are normalized for stable DQN training.
    """
    def __init__(self, dfs, batch_size=32, initial_balance=10000, fee_pct=0.001):
        super(BatchStockEnv, self).__init__()
        self.dfs = dfs
        self.n_stocks = len(dfs)
        self.batch_size = batch_size
        self.initial_balance = initial_balance
        self.fee_pct = fee_pct
        self.max_steps = min(len(df) for df in dfs) - 1

        # Action space: Buy=1, Hold=0, Sell=2 for each stock
        self.action_space = spaces.MultiDiscrete([3] * self.n_stocks)

        # Observation space: prices, ma5, ma10, momentum + balance + shares held
        self.obs_dim = self.n_stocks * 4 + 1 + self.n_stocks
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.obs_dim,), dtype=np.float32
        )

        # Precompute prices & moving averages
        self.prices = np.column_stack([df['Close'].values for df in dfs])
        self.ma5 = np.column_stack([df['Close'].rolling(5, min_periods=1).mean().values for df in dfs])
        self.ma10 = np.column_stack([df['Close'].rolling(10, min_periods=1).mean().values for df in dfs])

    def reset(self):
        self.current_step = np.random.randint(1, self.max_steps, size=self.batch_size)
        self.balance = np.full(self.batch_size, self.initial_balance, dtype=float)
        self.shares_held = np.zeros((self.batch_size, self.n_stocks), dtype=int)
        self.networth = self.balance.copy()
        return self._get_obs()

    def _get_obs(self):
        batch_obs = []
        for b in range(self.batch_size):
            step = max(1, self.current_step[b])

            # Flatten stock-related arrays
            prices = self.prices[step] / 1000.0
            ma5 = self.ma5[step] / 1000.0
            ma10 = self.ma10[step] / 1000.0
            momentum = (self.prices[step] - self.prices[step - 1]) / 100.0

            # Balance and shares scaled
            balance = np.array([self.balance[b] / (self.initial_balance * 10)], dtype=np.float32)
            shares = (self.shares_held[b] / 1000.0).astype(np.float32)

            obs = np.concatenate([prices, ma5, ma10, momentum, balance, shares]).astype(np.float32)
            batch_obs.append(obs)

        return np.stack(batch_obs, axis=0)

    def step(self, actions):
        rewards = np.zeros(self.batch_size)
        done = np.zeros(self.batch_size, dtype=bool)
        prev_values = self.shares_held * self.prices[self.current_step]

        # Process buy/sell
        for i in range(self.n_stocks):
            buy_mask = actions[:, i] == 1
            sell_mask = actions[:, i] == 2
            price_i = self.prices[self.current_step, i]

            max_shares = np.floor(self.balance / price_i).astype(int)
            buy_shares = np.where(buy_mask, max_shares, 0)
            cost = buy_shares * price_i * (1 + self.fee_pct)
            self.balance -= cost
            self.shares_held[:, i] += buy_shares
            rewards -= np.where(buy_mask & (max_shares == 0), 0.001, 0)  # small penalty

            sell_shares = np.where(sell_mask, self.shares_held[:, i], 0)
            gain = sell_shares * price_i * (1 - self.fee_pct)
            self.balance += gain
            self.shares_held[:, i] -= sell_shares
            rewards -= np.where(sell_mask & (sell_shares == 0), 0.001, 0)

        # Update net worth
        stock_values = self.shares_held * self.prices[self.current_step]
        new_networth = self.balance + stock_values.sum(axis=1)

        # Normalized rewards
        rewards += np.clip((new_networth - self.networth) / (self.networth + 1e-6), -0.05, 0.05)
        per_stock_rewards = np.clip((stock_values - prev_values) / (prev_values + 1e-6), -0.05, 0.05)
        rewards += per_stock_rewards.mean(axis=1)

        self.networth = new_networth

        # Step forward
        self.current_step += 1
        done[self.current_step >= self.max_steps] = True
        self.current_step = np.minimum(self.current_step, self.max_steps)

        return self._get_obs(), rewards, done, {}
