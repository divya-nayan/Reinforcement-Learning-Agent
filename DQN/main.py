import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from src.db_connection import SalesDatabase
from src.evaluation import DQNAgentEvaluator
from src.training import DQNTrainer
from src.utils import prepare_rl_dataset_new

#import these parameters as per your convenience.
server = server
database = database
username = username
password = password
port = port
sql_filepath = sql_filepath


if __name__ == "__main__":

    # Load the dataset (replace with actual file path)
    data = SalesDatabase().get_historical_sales_data(server, database, username, password, port, sql_filepath)
    standardized_data = prepare_rl_dataset_new(data)
    
    df_sorted = standardized_data.sort_values(by=['Year', 'Month', 'Day'])  # Sort by date
    train_data = df_sorted.groupby('RouteCode').head(900).reset_index(drop=True)
    test_data = df_sorted.groupby('RouteCode').tail(100).reset_index(drop=True)
    final_test_data = test_data.drop(columns=['ActualQuantitySold'])
    # Add a 'Padding' column with all zeros
    final_test_data['Padding'] = 0 


    # Calling DQNTrainer for training the DQNAgent on training data
    trained_agent, training_rewards = DQNTrainer(df=train_data, episodes=100).train()
    

    # Plot training rewards
    plt.plot(training_rewards)
    plt.title("Training Rewards Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

    # Evaluate the trained agent on testing data
    action_per_state, evaluation_df = DQNAgentEvaluator(agent=trained_agent, df=final_test_data).evaluate()
    evaluation_df['ActualQuantitySold'] = test_data['ActualQuantitySold']
    evaluation_df.drop(columns=['Padding'], inplace=True)
    evaluation_df.to_csv('states_and_actions.csv', index=False)

    # Plot evaluation rewards
    plt.plot(action_per_state)
    plt.title("Action Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Action")
    plt.show()
