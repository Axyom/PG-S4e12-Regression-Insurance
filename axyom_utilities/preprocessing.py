import numpy as np
import pandas as pd

def preprocess(df_in):
    df = df_in.copy()

    categorical_features = df.select_dtypes(include=['object', 'category']).columns
    
    df[categorical_features] = df[categorical_features].fillna("Unknown")

    for col in categorical_features:
        df[col] = df[col].astype('category')
    
    df["Policy Start Date"] = pd.to_datetime(df["Policy Start Date"])
    df["Month"]       = df["Policy Start Date"].dt.month
    df["Day"]         = df["Policy Start Date"].dt.day
    df["Week"]        = df["Policy Start Date"].dt.isocalendar().week
    df["Weekday"]     = df["Policy Start Date"].dt.weekday
    df['DaySin']      = np.sin(2 * np.pi * df['Day'] / 30)  
    df['DayCos']      = np.cos(2 * np.pi * df['Day'] / 30)
    df['WeekdaySin']  = np.sin(2 * np.pi * df['Weekday'] / 7)
    df['WeekdayCos']  = np.cos(2 * np.pi * df['Weekday'] / 7)
    
    df['DaysSinceStart']  = \
    np.ceil(
        (pd.to_datetime("12-31-2024") - df["Policy Start Date"])/ pd.Timedelta(1, "d")
    )

    df = df.drop("Policy Start Date", axis=1, errors = "ignore")
    

    return df