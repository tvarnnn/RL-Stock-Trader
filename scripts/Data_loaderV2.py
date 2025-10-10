import yfinance as yf
import pandas as pd

class MultiStockDataLoader:
    def __init__(self, tickers, start="2015-01-01", end="2025-01-01"):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.data = {}  # Store DataFrames keyed by ticker

    def download_data(self):
        for t in self.tickers:
            self.data[t] = yf.download(t, start=self.start, end=self.end)
        return self.data

    def get_data(self, ticker=None):  # If ticker specified, return that DataFrame
        if ticker:
            if ticker not in self.data:
                self.download_data()
            return self.data[ticker]
        if not self.data:  # Return all
            return self.download_data()
        return self.data

    def get_train_test(self, split=0.8):  # Split data 80/20
        train_data = {}
        test_data = {}
        for t, df in self.get_data().items():
            split_idx = int(len(df) * split)
            train_data[t] = df.iloc[:split_idx]
            test_data[t] = df.iloc[split_idx:]
        return train_data, test_data
