import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # Number of inputs features per observation
        self.action_size = action_size  # Number of possible actions
        self.memory = deque(maxlen=3000)  # Replay buffer to store memory
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # How fast epsilon decays per episode
        self.learning_rate = 0.005  # Learning rate

        # Main network
        self.model = self._build_model()
        # Target network for stability
        self.target_model = self._build_model()
        self.update_target_model()

    # Neural network to approximate Q-values
    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))  # 64 neurons
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    # Copy weights from main model to target model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Store experiences in replay buffer
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Choose action based on epsilon-greedy policy
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(act_values[0])

    # Train on a batch of experiences
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        # Prepare batches
        states = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.action_size))

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            target = self.model.predict(state[np.newaxis, :], verbose=0)[0]

            if done:
                target[action] = reward
            else:
                # Double DQN: select action with main model, evaluate with target model
                next_action = np.argmax(self.model.predict(next_state[np.newaxis, :], verbose=0)[0])
                target_val = self.target_model.predict(next_state[np.newaxis, :], verbose=0)[0][next_action]
                target[action] = reward + self.gamma * target_val

            targets[i] = target

        # Train on the entire batch at once
        self.model.fit(states, targets, epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
