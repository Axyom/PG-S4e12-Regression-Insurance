import numpy as np
import pandas as pd


def freq_encode(df, col, drop_org=False):
    """
    Detects categorical columns (str, category, object),
    applies frequency encoding, and updates the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        drop_org (bool): Whether to drop the original categorical columns.

    Returns:
        pd.DataFrame: Updated DataFrame with frequency-encoded columns.
    """
    # Detect categorical columns
    # cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    
    # for col in cat_cols:
        # Calculate frequency encoding
    freq_encoding = df[col].value_counts().to_dict()
    
    # Apply frequency encoding
    df[f"{col}_freq"] = df[col].map(freq_encoding).astype('float')
    
    # Drop the original column if specified
    if drop_org:
        df.drop(columns=[col], inplace=True)

    return df


def health_score_eng(df):
    df['HealthScore'] = df['Health Score'].apply(lambda x: int(x) if pd.notna(x) else x)
    df['Health Score'] = df['Health Score'].fillna('None').astype('string')
    
    return df

def preprocess(df_in):
    df = df_in.copy()

    df = clean_categorical(df)
    df = preprocess_dates(df)
    
    return df

def clean_categorical(df):
    categorical_features = df.select_dtypes(include=['object', 'category']).columns
    
    df[categorical_features] = df[categorical_features].fillna("Unknown")

    for col in categorical_features:
        df[col] = df[col].astype('category')   
    
    return df 

def all_to_string(df): 
    columns_to_convert = df.columns.difference(['Premium Amount'])
    df[columns_to_convert] = df[columns_to_convert].fillna('None').astype('string')
    
    return df

def preprocess_dates(df):
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

def reduce_memory_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if not pd.api.types.is_categorical_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            elif str(col_type)[:5] == 'float':
                # Avoid downcasting floats to float16
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    return df


def frequency_encode(train, test, drop_org=False):
    """
    Automatically detects categorical columns (str, category, object),
    applies frequency encoding, and updates train and test DataFrames.

    Parameters:
        train (pd.DataFrame): Training DataFrame.
        test (pd.DataFrame): Test DataFrame.
        drop_org (bool): Whether to drop the original categorical columns.

    Returns:
        tuple: (train, test)
    """
    # Detect categorical columns
    cat_cols = train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    
    # Combine train and test to calculate frequencies
    combined = pd.concat([train, test], axis=0, ignore_index=True)

    for col in cat_cols:
        freq_encoding = combined[col].value_counts().to_dict()
        
        # Apply frequency encoding
        train[f"{col}_freq"] = train[col].map(freq_encoding).astype('float')
        test[f"{col}_freq"] = test[col].map(freq_encoding).astype('float')
        
        # Drop the original column if specified
        if drop_org:
            train.drop(columns=[col], inplace=True)
            test.drop(columns=[col], inplace=True)

    return train, test
  