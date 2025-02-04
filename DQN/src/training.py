import numpy as np
from src.environment import SalesEnvironment
from src.agent import DQNAgent

# from db_connection import SalesDatabase
# from utils import prepare_rl_dataset
# data = SalesDatabase().get_historical_sales_data()
# df = prepare_rl_dataset(data)

class DQNTrainer:
    
    def __init__(self, df, episodes, batch_size=32, max_steps=50):
        self.df = df
        self.env = SalesEnvironment(df)  # Initialize the environment
        self.state_size = len(self.env.state_space)  # Number of features in the state
        self.action_size = len(self.env.action_space)  # Number of possible actions
        self.agent = DQNAgent(self.state_size, self.action_size)  # Initialize the agent
        self.episodes = episodes  # Number of training episodes
        self.batch_size = batch_size  # Batch size for experience replay
        self.max_steps = max_steps  # Maximum steps per episode
        self.rewards = []  # To store the rewards per episode


    def _process_state(self, state):
        """
        Helper function to process the state into a numpy array.
        """
        state_dict = state.item()  # Extract the dictionary from the array
        state_array = np.array([
                state_dict['Day'],
                state_dict['Month'],
                state_dict['Year'],
                state_dict['WeekDay'],
                state_dict['RouteCode'],
                state_dict['ItemCode'],
                state_dict['ActualQuantitySold'],
                state_dict['PredictedQuantity'],
                state_dict['UnitPrice'],
                state_dict['SalesRollingAvg'],
                state_dict['BadReturnRollingAvg'],
                state_dict['ActualQuantityRollingAvg']
            ], dtype=float)
        return np.reshape(state_array, [1, len(state_array)])  # Reshape state array for DQN input


    def _process_next_state(self, next_state):
        """
        Helper function to process the next state into a numpy array.
        """
        next_state_dict = next_state.item()  # Extract the dictionary from the array
        next_state_array = np.array([
                next_state_dict['Day'],
                next_state_dict['Month'],
                next_state_dict['Year'],
                next_state_dict['WeekDay'],
                next_state_dict['RouteCode'],
                next_state_dict['ItemCode'],
                next_state_dict['ActualQuantitySold'],
                next_state_dict['PredictedQuantity'],
                next_state_dict['UnitPrice'],
                next_state_dict['SalesRollingAvg'],
                next_state_dict['BadReturnRollingAvg'],
                next_state_dict['ActualQuantityRollingAvg']
            ], dtype=float)
        return np.reshape(next_state_array, [1, len(next_state_array)]) 


    def train(self):
        # Training Loop

        for episode in range(self.episodes):
            total_reward = 0  
            
            # Reset the environment for a new episode
            state = self.env.reset()
            state = self._process_state(state)

            for step in range(self.max_steps):
                action = self.agent.act(state)
                next_state, reward, done = self.env.step(action)

                next_state = self._process_next_state(next_state)
                
                # Store the experience in the agent's memory
                self.agent.remember(state, action, reward, next_state)

                state = next_state
                total_reward += reward

                # Replay the experience if memory size exceeds the batch size
                if len(self.agent.memory) > self.batch_size:
                    self.agent.replay(self.batch_size)

                if done:
                    break

            self.rewards.append(total_reward)

            # Print progress for each episode
            print(f"Episode {episode + 1}/{self.episodes}, Total Reward: {total_reward}, Epsilon: {self.agent.epsilon:.2f}")

        return self.agent, self.rewards
    