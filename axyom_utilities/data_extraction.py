import os
import pandas as pd
import numpy as np

def extract_data(data_dir, use_original_data):
    
    # Construct file paths
    train_file = os.path.join(data_dir, "train.csv")
    test_file = os.path.join(data_dir, "test.csv")

    # Load the datasets
    train_df = pd.read_csv(train_file, index_col="id")
    test_df = pd.read_csv(test_file, index_col="id")
    
    X_train = train_df.drop('Premium Amount', axis=1)
    y_train = pd.DataFrame(np.log1p(train_df['Premium Amount'].values)) # Log Space
    X_test = test_df

    if use_original_data:
        original_file = os.path.join(data_dir, "Insurance Premium Prediction Dataset.csv")
        original_df = pd.read_csv(original_file).dropna(subset=["Premium Amount"])
        X_train["Synthetic"] = 1
        X_test["Synthetic"] = 1
        X_orig = original_df.drop('Premium Amount', axis=1)
        X_orig["Synthetic"] = 0
        y_orig = pd.DataFrame(np.log1p(original_df['Premium Amount'].values))
        
    else:
        X_orig = None
        y_orig = None
        
    return X_train, y_train, X_test, X_orig, y_orig