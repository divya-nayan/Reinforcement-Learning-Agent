import numpy as np
import random


# Define the Sales Environment for Reinforcement Learning
class SalesEnvironment:
    def __init__(self, df):
        """
        Reinforcement Learning environment for sales optimization.
        """
        self.states = df
        self.max_sales = 1000  # Prevent division by zero
        self.max_demand = df['ActualQuantitySold'].max() + 1e-8
        self.state_space = self.states.iloc[0]
        self.action_space = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Actions: [no change, 5%-20% increment/decrement]
        self.history_length = 7  # Rolling window for trends

        # Initialize tracking lists for early stopping
        self.recent_rewards = []
        self.recent_actions = []

    def reset(self):
        """ Resets the environment and returns the initial state. """
        self.current_index = np.random.randint(0, len(self.states) - self.history_length)
        self.current_state = self.get_state(self.current_index)
        return np.array(self.current_state)

    def step(self, action):
        """
        Executes an action, updates state, and calculates reward.
        """
        state = self.get_state(self.current_index)
        # Convert dictionary values to a tuple
        state_tuple = tuple(state.values())

        # Unpack state variables
        (Day, Month, Year, WeekDay, RouteCode, ItemCode, ActualQuantitySold,
        PredictedQuantity, UnitPrice, SalesRollingAvg, BadReturnRollingAvg, ActualQuantityRollingAvg) = state_tuple

        # Action adjustment factors
        adjustment_factors = {0: 0, 1: 0.05, 2: 0.10, 3: 0.15, 4: 0.20,  
                               5: -0.05, 6: -0.10, 7: -0.15, 8: -0.20}
        adjustment_percentage = adjustment_factors[action]

        # Adjust predicted quantity
        adjusted_predicted_quantity = max(0, PredictedQuantity * (1 + adjustment_percentage))

        # Simulate demand variation (market fluctuation)
        demand_variation = np.random.normal(loc=0, scale=0.1 * ActualQuantitySold)
        new_actual_quantity_sold = max(0, ActualQuantitySold + demand_variation)

        # Compute sales and bad return values
        total_sales_value = min(adjusted_predicted_quantity, new_actual_quantity_sold) * UnitPrice
        total_bad_return_value = max(0, adjusted_predicted_quantity - new_actual_quantity_sold) * UnitPrice

        # Update rolling averages using exponential moving average
        alpha = 2 / (self.history_length + 1)  # Smoothing factor
        new_sales_rolling_avg = alpha * total_sales_value + (1 - alpha) * SalesRollingAvg
        new_bad_return_rolling_avg = alpha * total_bad_return_value + (1 - alpha) * BadReturnRollingAvg
        new_actual_quantity_rolling_avg = alpha * new_actual_quantity_sold + (1 - alpha) * ActualQuantityRollingAvg

        # Prepare next state
        next_state = {
            'Day': Day,
            'Month': Month,
            'Year': Year,
            'WeekDay': WeekDay,
            'RouteCode': RouteCode,
            'ItemCode': ItemCode,
            'ActualQuantitySold': ActualQuantitySold,
            'PredictedQuantity': adjusted_predicted_quantity,
            'UnitPrice': UnitPrice,
            'SalesRollingAvg': new_sales_rolling_avg,
            'BadReturnRollingAvg': new_bad_return_rolling_avg,
            'ActualQuantityRollingAvg': new_actual_quantity_rolling_avg
        }
        next_state = np.array(next_state)

        # Compute reward
        reward = self.calculate_reward(total_bad_return_value, total_sales_value, 
                                       new_actual_quantity_sold, adjusted_predicted_quantity)


        # Check early stopping criteria
        done = self.should_stop_early()

        return next_state, reward, done

    def calculate_reward(self, total_bad_return_value, total_sales_value, actual_sold, predicted_quantity):
        """
        Reward function that:
        - Maximizes sales value
        - Minimizes waste (bad returns)
        - Encourages accurate predictions
        """
        profit = total_sales_value - total_bad_return_value
        waste_ratio = total_bad_return_value / (total_sales_value + total_bad_return_value + 1e-8)
        accuracy_bonus = 1.0 - waste_ratio  # 1 when no waste, 0 when high waste
        demand_supply_balance = 1 - abs(predicted_quantity - actual_sold) / (actual_sold + 1e-8)

        reward = (
            0.6 * (profit / self.max_sales) +  # Encourage high profit
            0.3 * accuracy_bonus +             # Encourage minimal waste
            0.1 * demand_supply_balance        # Encourage balanced predictions
        )

        # Penalize drastic changes in predicted sales
        change_penalty = abs(predicted_quantity - actual_sold) / (actual_sold + 1e-8)
        reward -= 0.1 * change_penalty  # Penalty for erratic predictions

        return np.clip(reward, -2, 2)  # Clip the reward to the range [-2, 2] 

    def should_stop_early(self):
        """
        Early stopping condition to prevent unnecessary training if:
        - Reward is stable
        - Actions are converging
        - Minimal state change is observed
        """
        if len(self.recent_rewards) > 50:
            avg_reward = np.mean(self.recent_rewards[-50:])
            reward_variation = np.std(self.recent_rewards[-50:])
            
            if avg_reward > 1.8:  # If reward is already optimal
                return True
            
            if reward_variation < 0.05:  # If reward has stabilized
                return True
        
        if len(self.recent_actions) > 50:
            # Check for convergence in recent actions
            if np.mean(np.array(self.recent_actions[-10:]) == self.recent_actions[-1]) > 0.8:
                return True  # Actions are repeating, indicating convergence
        
        return False  # Continue training

    def get_state(self, index):
        """ Retrieves a single state from the dataset based on index. """
        self.current_state = self.states.iloc[index].to_dict()
        return (self.current_state)
