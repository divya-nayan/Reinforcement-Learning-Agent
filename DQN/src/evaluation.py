import numpy as np
import pandas as pd
from src.environment import SalesEnvironment
from src.agent import DQNAgent




class DQNAgentEvaluator:

    def __init__(self, agent, df):
        """
        Initializes the evaluator with the agent, dataset, and number of episodes.

        Args:
        - agent: The trained DQN agent.
        - df: The dataset (dataframe).
        - episodes: Number of episodes to evaluate the agent.
        - max_steps: Maximum steps per episode (default: 100).
        """
        self.agent = agent  # The trained agent
        self.df = df  # The dataset

    def _process_state(self, state):
        """
        Helper function to process the state into a numpy array.

        Args:
        - state: The state object to process.

        Returns:
        - Processed state as a numpy array.
        """
        # state_dict = state.item()
        state_array = np.array([
                state['Day'],
                state['Month'],
                state['Year'],
                state['WeekDay'],
                state['RouteCode'],
                state['ItemCode'],
                state['PredictedQuantity'],
                state['UnitPrice'],
                state['SalesRollingAvg'],
                state['BadReturnRollingAvg'],
                state['ActualQuantityRollingAvg'],
                state['Padding']
            ], dtype=float)
        return np.reshape(state_array, [1, len(state_array)])  # Reshape state for DQN input
    

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
                next_state_dict['PredictedQuantity'],
                next_state_dict['UnitPrice'],
                next_state_dict['SalesRollingAvg'],
                next_state_dict['BadReturnRollingAvg'],
                next_state_dict['ActualQuantityRollingAvg'],
                next_state_dict['Padding']
            ], dtype=float)
        return np.reshape(next_state_array, [1, len(next_state_array)]) 


    def evaluate(self):
        """
        Evaluates the agent for all states in the dataset and returns:
        - DataFrame containing states, predicted Q-values, and the best action.

        Returns:
        - df_all_best_actions: DataFrame with all states and their predicted best actions.
        """
        all_states_best_actions = []  # To store results for all states
        action_per_state = []  # To store the action taken for each state

        for index, row in self.df.iterrows(): 
            state = self._process_state(row)
            q_values = self.agent.model.predict(state, verbose=0)[0]
            action = np.argmax(q_values)
            action_per_state.append(action)
            state_flat = row.values.flatten()  
            # Store the state, best action, and the maximum reward
            all_states_best_actions.append(
                list(state_flat) + [action]
            )
            print(f"Action taken over {index} episodes: {action}")
        # Define column names based on the dataset and the additional fields
        column_names = list(self.df.columns) + ['BestAction']
        df_all_best_actions = pd.DataFrame(all_states_best_actions, columns=column_names)

        return action_per_state ,df_all_best_actions



