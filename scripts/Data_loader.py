import yfinance as yf
import pandas as pd

class StockDataLoader:
    def __init__(self, ticker, start="2015-01-01", end="2025-01-01"):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.data = None

    def download_data(self):
        self.data = yf.download(self.ticker, start = self.start, end = self.end)
        return self.data

    def get_data(self): # Download the data first, otherwise, return the data if its already been downloaded
        if self.data is None:
            return self.download_data()
        return self.data

    def get_train_test(self, split=0.8): #Splitting the data 80/20
        df = self.get_data() # Grab full dataset
        split_idx = int(len(df) * split) #Splitting the dataset 80/20
        train = df.iloc[:split_idx]  # Calculate 80% of data for training
        test = df.iloc[split_idx:] # Leave the remainder of the dataset for testing
        return train, test
