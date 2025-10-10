RL-Stock-Trader

RL-Stock-Trader is a deep reinforcement learning (RL) program that uses a Deep Q-Network (DQN) agent to learn trading strategies for multiple stocks. The agent interacts with a custom multi-stock environment, learning to maximize portfolio net worth over time.

Features:

Multi-stock trading simulation with adjustable tickers.

DQN agent with epsilon-greedy and softmax action selection strategies.

Risk-adjusted and allocation-aware reward system.

Replay buffer and target network for stable RL training.

Train/test evaluation and net worth visualization over episodes.

Supports GPU acceleration with PyTorch.

Note: This repository includes multiple iterations of the DQN agent, data loader, and stock environment (e.g., V1, V2, V3). The latest versions (DQNAgentV3, StockEnvV3, Data_loaderV2) are actively used in main.py.


Technologies:

Python 3.10+

PyTorch

Gymnasium

Matplotlib

Pandas

yFinance (for stock data)


Installation: 

Clone the repository:

git clone https://github.com/yourusername/RL-Stock-Trader.git
cd RL-Stock-Trader


Create a virtual environment (optional but HEAVILY recommended):

python -m venv venv
venv\Scripts\activate 


Install dependencies:

pip install -r requirements.txt

Usage

Open main.py and specify the stock tickers you want to trade:

tickers = ["TSLA", "AAPL", "AMZN"]


Run the script:

python main.py


The program will train the DQN agent for the specified number of episodes and display a plot of Train vs Test Net Worth.

Project Structure
RL-Stock-Trader/
|-- main.py                 # Main training script
|-- scripts/
    |--- Data_loader.py 
    |--- Data_loaderV2.py    # Latest stock data loader (older version available
|-- envs/
    |--- stock_env.py
    |--- stock_envV2.py
    |--- stock_envV3.py
    |--- stock_envV4.py      # Latest multi-stock trading environment (older versions V1-V3 available)
|-- models/
    |--- dqn_agent.py
    |--- dqn_agentV2.py
│   |--- dqn_agentV3.py      # Latest DQN agent implementation (older versions V1-V2 available)
|-- requirements.txt         # Python dependencies
|-- README.md
