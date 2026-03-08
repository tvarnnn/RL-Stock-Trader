import gymnasium as gym
from gymnasium import spaces
import numpy as np


def _compute_rsi(prices_col, period=14):
    """
    Wilder's RSI for a 1-D price array.
    Uses exponential smoothing (com = period-1) via a manual loop to avoid
    pandas dependency inside the env. Returns values in [0, 1].
    """
    T     = len(prices_col)
    delta = np.diff(prices_col.astype(np.float64), prepend=prices_col[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.zeros(T, dtype=np.float64)
    avg_loss = np.zeros(T, dtype=np.float64)

    if T > period:
        avg_gain[period] = gain[1:period + 1].mean()
        avg_loss[period] = loss[1:period + 1].mean()
        for i in range(period + 1, T):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    rs  = avg_gain / (avg_loss + 1e-8)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return (rsi / 100.0).astype(np.float32)   # normalized to [0, 1]


class BatchStockEnv(gym.Env):
    """
    Vectorized multi-stock trading environment.
    Observations include: price, MA5, MA10, momentum, RSI, log-volume (per stock)
    plus portfolio balance and shares held.
    All observation construction is fully numpy-vectorized (no Python loops).
    """
    def __init__(self, dfs, batch_size=32, initial_balance=10000, fee_pct=0.001):
        super().__init__()
        self.dfs             = dfs
        self.n_stocks        = len(dfs)
        self.batch_size      = batch_size
        self.initial_balance = initial_balance
        self.fee_pct         = fee_pct
        self.max_steps       = min(len(df) for df in dfs) - 1

        # Action space: Hold=0, Buy=1, Sell=2 per stock
        self.action_space = spaces.MultiDiscrete([3] * self.n_stocks)

        # Observations: 6 signals per stock (price, ma5, ma10, momentum, rsi, volume)
        #               + 1 balance + n_stocks shares
        self.obs_dim = self.n_stocks * 6 + 1 + self.n_stocks
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # ── Precompute all arrays once at init ────────────────────────────────
        # Price & moving averages  shape: (T, n_stocks)
        self.prices = np.column_stack([df['Close'].values for df in dfs]).astype(np.float32)
        self.ma5    = np.column_stack(
            [df['Close'].rolling(5,  min_periods=1).mean().values for df in dfs]
        ).astype(np.float32)
        self.ma10   = np.column_stack(
            [df['Close'].rolling(10, min_periods=1).mean().values for df in dfs]
        ).astype(np.float32)

        # Momentum: one-step price change  (step 0 gets zero by prepend)
        self.momentum = np.diff(self.prices, axis=0, prepend=self.prices[[0]]).astype(np.float32)

        # RSI per stock  shape: (T, n_stocks)
        self.rsi = np.column_stack(
            [_compute_rsi(self.prices[:, i]) for i in range(self.n_stocks)]
        ).astype(np.float32)

        # Log-volume, z-score normalized per stock  shape: (T, n_stocks)
        raw_vol  = np.column_stack([df['Volume'].values for df in dfs]).astype(np.float64)
        log_vol  = np.log1p(raw_vol)
        vol_mean = log_vol.mean(axis=0)
        vol_std  = log_vol.std(axis=0) + 1e-8
        self.volumes = ((log_vol - vol_mean) / vol_std).astype(np.float32)

        # Fixed scaling constants for non-normalized features
        self._price_scale   = np.float32(1000.0)
        self._mom_scale     = np.float32(100.0)
        self._balance_scale = np.float32(self.initial_balance * 10)
        self._share_scale   = np.float32(1000.0)

    def reset(self):
        self.current_step = np.random.randint(1, self.max_steps, size=self.batch_size)
        self.balance      = np.full(self.batch_size, self.initial_balance, dtype=np.float32)
        self.shares_held  = np.zeros((self.batch_size, self.n_stocks), dtype=np.float32)
        self.networth     = self.balance.copy()
        return self._get_obs()

    def _get_obs(self):
        steps = np.maximum(self.current_step, 1)   # (batch,)

        prices   = self.prices[steps]   / self._price_scale    # (batch, n_stocks)
        ma5      = self.ma5[steps]      / self._price_scale
        ma10     = self.ma10[steps]     / self._price_scale
        momentum = self.momentum[steps] / self._mom_scale
        rsi      = self.rsi[steps]                             # already in [0, 1]
        volume   = self.volumes[steps]                         # z-scored log-volume
        balance  = (self.balance / self._balance_scale).reshape(-1, 1)
        shares   = self.shares_held / self._share_scale

        return np.concatenate([prices, ma5, ma10, momentum, rsi, volume, balance, shares], axis=1)

    def step(self, actions):
        rewards     = np.zeros(self.batch_size, dtype=np.float32)
        prev_values = self.shares_held * self.prices[self.current_step]

        for i in range(self.n_stocks):
            buy_mask  = actions[:, i] == 1
            sell_mask = actions[:, i] == 2
            price_i   = self.prices[self.current_step, i]   # (batch,)

            # Buy: spend as much balance as possible
            max_shares = np.floor(self.balance / price_i).astype(np.float32)
            buy_shares = np.where(buy_mask, max_shares, 0.0)
            cost = buy_shares * price_i * (1 + self.fee_pct)
            self.balance          -= cost
            self.shares_held[:, i] += buy_shares
            rewards -= np.where(buy_mask & (max_shares == 0), 0.001, 0.0)

            # Sell: liquidate all held shares
            sell_shares = np.where(sell_mask, self.shares_held[:, i], 0.0)
            gain = sell_shares * price_i * (1 - self.fee_pct)
            self.balance          += gain
            self.shares_held[:, i] -= sell_shares
            rewards -= np.where(sell_mask & (sell_shares == 0), 0.001, 0.0)

        # Net worth delta reward
        stock_values = self.shares_held * self.prices[self.current_step]
        new_networth = self.balance + stock_values.sum(axis=1)

        rewards += np.clip((new_networth - self.networth) / (self.networth + 1e-6), -0.05, 0.05)
        per_stock_gain = np.clip((stock_values - prev_values) / (prev_values + 1e-6), -0.05, 0.05)
        rewards += per_stock_gain.mean(axis=1)

        self.networth = new_networth

        self.current_step += 1
        done = self.current_step >= self.max_steps
        self.current_step = np.minimum(self.current_step, self.max_steps)

        return self._get_obs(), rewards, done, {}
