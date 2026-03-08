import yfinance as yf


class MultiStockDataLoader:
    def __init__(self, tickers, start="2015-01-01", end="2025-01-01"):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.data = {}

    def download_data(self):
        # Single batch request instead of N sequential downloads
        raw = yf.download(self.tickers, start=self.start, end=self.end, group_by="ticker", auto_adjust=True)
        for t in self.tickers:
            self.data[t] = raw[t].dropna()
        return self.data

    def get_data(self, ticker=None):
        if not self.data:
            self.download_data()
        if ticker:
            return self.data[ticker]
        return self.data

    def get_train_test(self, split=0.8):
        train_data, test_data = {}, {}
        for t, df in self.get_data().items():
            idx = int(len(df) * split)
            train_data[t] = df.iloc[:idx]
            test_data[t]  = df.iloc[idx:]
        return train_data, test_data
