import numpy as np
import random

class SalesEnvironment:
    def __init__(self, df):
        """
        Reinforcement Learning environment for sales optimization.
        """
        self.states = df
        self.state_space = self.states.iloc[0]
        # Actions: [no change, 5%-20% increment/decrement]
        self.action_space = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.history_length = 30  # Rolling window for trends

        # Tracking lists for early stopping criteria
        self.recent_rewards = []
        self.recent_actions = []

    def reset(self):
        """Resets the environment and returns the initial state."""
        self.recent_rewards = []
        self.recent_actions = []
        self.current_index = np.random.randint(0, len(self.states) - self.history_length)
        self.current_state = self.get_state(self.current_index)
        return np.array(self.current_state)

    def step(self, action):
        """
        Executes an action, updates the state, and calculates reward.
        Here, the reward is based solely on the improvement (or deterioration)
        in the rolling averages.
        """
        # Get the current state as a dictionary
        state = self.get_state(self.current_index)
        # Convert dictionary values to a tuple (ensure a consistent ordering)
        state_tuple = tuple(state.values())

        # Unpack state variables (order must be consistent with your data)
        (Day_sin, Day_cos, Month_sin, Month_cos, Year_sin, Year_cos,
         WeekDay_sin, Weekday_cos, RouteCode, ItemCode, ActualQuantitySold,
         PredictedQuantity, UnitPrice, SalesRollingAvg, TotalReturnRollingAvg,
         ActualQuantityRollingAvg) = state_tuple

        # Define action adjustment factors:
        # 0: no change, 1-4: positive adjustments, 5-8: negative adjustments.
        adjustment_factors = {
            0: 0, 1: 0.05, 2: 0.10, 3: 0.15, 4: 0.20,
            5: -0.05, 6: -0.10, 7: -0.15, 8: -0.20
        }
        adjustment_percentage = adjustment_factors[action]
        # Adjust predicted quantity based on action
        adjusted_predicted_quantity = max(0, PredictedQuantity * (1 + adjustment_percentage))


        # Compute immediate values (which we won’t use directly in the reward)
        total_sales_value = min(adjusted_predicted_quantity, ActualQuantitySold) * UnitPrice
        total_return_value = max(0, adjusted_predicted_quantity - ActualQuantitySold) * UnitPrice

        # Update rolling averages using an exponential moving average
        alpha = 2 / (self.history_length + 1)
        new_sales_rolling_avg = alpha * total_sales_value + (1 - alpha) * SalesRollingAvg
        new_total_return_rolling_avg = alpha * total_return_value + (1 - alpha) * TotalReturnRollingAvg
        
        # Build the next state as a dictionary and then convert to numpy array
        next_state = {
            'Day_sin': Day_sin,
            'Day_cos': Day_cos,
            'Month_sin': Month_sin,
            'Month_cos': Month_cos,
            'Year_sin': Year_sin,
            'Year_cos': Year_cos,
            'WeekDay_sin': WeekDay_sin,
            'WeekDay_cos': Weekday_cos,
            'RouteCode': RouteCode,
            'ItemCode': ItemCode,
            'ActualQuantitySold': ActualQuantitySold,
            'PredictedQuantity': adjusted_predicted_quantity,
            'UnitPrice': UnitPrice,
            'SalesRollingAvg': new_sales_rolling_avg,
            'TotalReturnRollingAvg': new_total_return_rolling_avg,
            'ActualQuantityRollingAvg': ActualQuantityRollingAvg
        }
        next_state = np.array(next_state)

        # Calculate the reward based solely on improvements in rolling metrics
        reward = self.calculate_reward(
            SalesRollingAvg, new_sales_rolling_avg,
            TotalReturnRollingAvg, new_total_return_rolling_avg,
            ActualQuantitySold, adjusted_predicted_quantity
        )

        # Record the action and reward for early stopping checks
        self.recent_actions.append(action)
        self.recent_rewards.append(reward)
        done = self.should_stop_early()

        return next_state, reward, done

    def calculate_reward(self, old_sales, new_sales,
                        old_return, new_return,
                        actual_quantity, adjusted_prediction):
        """
        Calculates reward based on the change in rolling averages.
          - An increase in SalesRollingAvg is rewarded.
          - A decrease in BadReturnRollingAvg is rewarded.
          - Changes in ActualQuantityRollingAvg can be rewarded (or penalized)
            based on your specific business objectives.
        """
        # Compute the improvements.
        delta_sales = new_sales - old_sales
        delta_returns = old_return - new_return
        delta_gap = adjusted_prediction - actual_quantity

        # Weight the improvements (tune these weights as needed).
        reward = (1 * delta_sales) + (0.5 * delta_returns) + (0.2 * delta_gap)

        return reward

    def should_stop_early(self):
        """
        Early stopping condition to prevent unnecessary training if:
          - Recent rewards are stable or optimal.
          - Recent actions have converged.
        """
        if len(self.recent_rewards) > 30:
            avg_reward = np.mean(self.recent_rewards[-25:])
            reward_variation = np.std(self.recent_rewards[-25:])
            reward_90_percentile = np.percentile(self.recent_rewards[-25:], 90)
            if avg_reward >= reward_90_percentile:
                return True
            if reward_variation < max(0.05 * abs(avg_reward), 0.01):
                return True

        if len(self.recent_actions) > 30:
            if len(set(self.recent_actions[-8:])) == 1:
                return True

        return False

    def get_state(self, index):
        """Retrieves a single state from the dataset based on the given index."""
        self.current_state = self.states.iloc[index].to_dict()
        return self.current_state

# Example usage:
# ex = SalesEnvironment(data)
# state = ex.reset()
# next_state, reward, done = ex.step(3)
# print(next_state, reward, done)
