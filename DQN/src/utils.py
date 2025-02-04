import pandas as pd
from sklearn.preprocessing import LabelEncoder

# from db_connection import SalesDatabase
# sales_db = SalesDatabase()  # No need to pass any parameters
# df = sales_db.get_historical_sales_data()



def prepare_rl_dataset(df):
    """
    Prepare the RL dataset with required columns, including mapping WeekDay to numerical values,
    efficiently handling NaN values, and ensuring valid calculations. Adds rolling averages for the last 7 days.

    Args:
        df (pd.DataFrame): Input dataframe containing the raw data.

    Returns:
        pd.DataFrame: Transformed dataframe with required columns, clipped values,
                      and efficiently calculated features.
    """
    # Rename relevant columns
    df = df.rename(columns={
        'day': 'Day',
        'month': 'Month',
        'year': 'Year',
        'day_of_week': 'WeekDay',
        'CustomerCode' : 'RouteItemCode',
        'ItemCode': 'ItemCode',
        'Actual': 'ActualQuantitySold',
        'Predicted': 'PredictedQuantity',
        'UnitPrice': 'UnitPrice',
        'Totalsalesvalue': 'TotalSalesValue',
        'Totalbadreturnvalue': 'TotalBadReturnValue',
    })

    # Map WeekDay column to numerical values
    weekday_mapping = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6
    }
    if 'WeekDay' in df.columns:
        df['WeekDay'] = df['WeekDay'].map(weekday_mapping)

    # Handle RouteItemCode: if it has only one category, set all values to 0
    if df['RouteItemCode'].nunique() == 1:
        df['RouteItemCode'] = 0

    # Replace NaN values with the mode of the respective column
    for column in df.columns:
        mode_value = df[column].mode()[0]  # Get the mode of the column
        df[column] = df[column].fillna(mode_value)  # Replace NaN with mode

    # Clip numerical columns to ensure valid values
    df['ActualQuantitySold'] = df['ActualQuantitySold'].clip(lower=0)
    df['UnitPrice'] = df['UnitPrice'].clip(lower=0)
    df['PredictedQuantity'] = df['PredictedQuantity'].clip(lower=0)
    df['TotalSalesValue'] = df['TotalSalesValue'].clip(lower=0)
    df['TotalBadReturnValue'] = df['TotalBadReturnValue'].clip(lower=0)

    # Sort by date to calculate rolling averages
    df = df.sort_values(by=['Year', 'Month', 'Day']).reset_index(drop=True)

    # Add rolling averages for the last 7 days
    df['SalesRollingAvg'] = (
        df['TotalSalesValue']
        .rolling(window=7, min_periods=1)
        .mean()
        .clip(lower=0)
    )
    df['BadReturnRollingAvg'] = (
        df['TotalBadReturnValue']
        .rolling(window=7, min_periods=1)
        .mean()
        .clip(lower=0)
    )

    # Select and reorder the required columns
    result_df = df[['Day', 'Month', 'Year', 'WeekDay', 'RouteItemCode', 'ActualQuantitySold',
                    'PredictedQuantity', 'UnitPrice', 'TotalBadReturnValue', 'TotalSalesValue',
                    'SalesRollingAvg', 'BadReturnRollingAvg']]

    return result_df




# state_data = prepare_rl_dataset(df)
# print(state_data.head())



def prepare_rl_dataset_new(df):
    """
    Prepare the RL dataset with lagged rolling averages per (RouteCode, ItemCode),
    efficient sorting, and minimal redundant operations.

    Args:
        df (pd.DataFrame): Input dataframe containing the raw data.

    Returns:
        pd.DataFrame: Transformed dataframe with required features for DQN.
    """
    # Rename columns
    df = df.rename(columns={
        'day': 'Day',
        'month': 'Month',
        'year': 'Year',
        'day_of_week': 'WeekDay',
        'CustomerCode': 'RouteCode',
        'ItemCode': 'ItemCode',
        'Actual': 'ActualQuantitySold',
        'Predicted': 'PredictedQuantity',
        'UnitPrice': 'UnitPrice',
        'Totalsalesvalue': 'TotalSalesValue',
        'Totalbadreturnvalue': 'TotalBadReturnValue',
    })

    # Convert weekday to numerical
    weekday_mapping = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, 
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    }
    df['WeekDay'] = df['WeekDay'].map(weekday_mapping)

    # ItemCode mapping
    itemcode_mapping = {
        "50-0261": 0, "50-4072": 1, "50-4085": 2, "50-0412": 3, 
        "50-0117": 4, "50-0276": 5, "50-0102": 6, "50-0526": 7, 
        "50-0401": 8, "50-0296": 9
    }
    df['ItemCode'] = df['ItemCode'].map(itemcode_mapping)

    # RouteCode mapping
    routecode_mapping = {
        "3003": 0, "3004": 1
    }
    df['RouteCode'] = df['RouteCode'].map(routecode_mapping)

    # Fill missing values with mode (optimized)
    df = df.apply(lambda col: col.fillna(col.mode()[0]) if col.isna().any() else col)

    # Clip numerical values
    num_cols = ['ActualQuantitySold', 'UnitPrice', 'PredictedQuantity']
    df[num_cols] = df[num_cols].clip(lower=0)

    # Sort once for all subsequent operations
    df = df.sort_values(by=['Year', 'Month', 'Day'])

    # Calculate **lagged** rolling averages (excluding yesterday and today)
    rolling_cols = {
        'ActualQuantitySold': 'ActualQuantityRollingAvg',
        'TotalSalesValue': 'SalesRollingAvg',
        'TotalBadReturnValue': 'BadReturnRollingAvg'
    }
    
    for col, new_col in rolling_cols.items():
        df[new_col] = (
            df.groupby(['RouteCode', 'ItemCode'])[col]
            .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
            .fillna(0)  # Shift before rolling
            .clip(lower=0)
        )

    # Select and order final columns
    final_cols = [
        'Day', 'Month', 'Year', 'WeekDay', 'RouteCode', 'ItemCode',
        'ActualQuantitySold', 'PredictedQuantity', 'UnitPrice', 
        'SalesRollingAvg', 'BadReturnRollingAvg', 'ActualQuantityRollingAvg'
    ]
    
    # Get top 60 records per RouteCode (already sorted by date)
    return (
        df[final_cols]
        .groupby('RouteCode')
        .head(1000)
        .reset_index(drop=True)
    )


# new_data = prepare_rl_dataset_new(df)
# print(new_data.shape)
# print(new_data.head())
# print(new_data.describe())