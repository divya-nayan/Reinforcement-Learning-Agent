import os
import tensorflow as tf
from collections import deque
import random
import numpy as np



# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, update_frequency=10):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.97  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.update_frequency = update_frequency  # Frequency of target model updates
        self.episode_count = 0  # Counter for episodes

        # Build the main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_dim=self.state_size),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(32),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss=tf.keras.losses.Huber())
        return model

    def update_target_model(self):
        """Update target model to match the weights of the main model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state):
        """Store the experience in memory."""
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        """Return action based on epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Exploration
        q_values = self.model.predict(state, verbose=0)  # Exploitation
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        """Train the model based on a batch of experiences."""
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward
            if next_state is not None:
                # Calculate the target Q-value using the target model
                target += self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])
            # Get the current Q-values from the main model
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target  # Update the Q-value for the taken action

            # Perform one training step
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # Decay epsilon to decrease exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update the target model periodically
        self.episode_count += 1
        if self.episode_count % self.update_frequency == 0:
            self.update_target_model()
