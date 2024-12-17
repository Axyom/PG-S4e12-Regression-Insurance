from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import numpy as np
import pandas as pd

def train_model_cv(model, X_train, y_train, X_test, X_orig, y_orig=None, cv_splits=7, early_stopping_rounds=None):
    # Initialize the K-Fold for CV
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=84)
    
    # Initialize placeholders for results
    oof_preds = np.zeros(X_train.shape[0])
    test_preds = np.zeros(X_test.shape[0])
    cv_scores = np.zeros(cv_splits)
    best_iterations = np.zeros(cv_splits)
    models = []
    
    # Loop through each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"Training fold {fold + 1}...")
        
        # Split data
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        if X_orig is not None:
            # Append rows
            X_train_fold = pd.concat([X_train_fold, X_orig], ignore_index=True)
            y_train_fold = pd.concat([y_train_fold, y_orig], ignore_index=True)
        
        # Fit the model on training data
        if early_stopping_rounds:
            model.fit(
                X_train_fold, y_train_fold, 
                eval_set=(X_val_fold, y_val_fold),
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
            best_iterations[fold]=model.get_best_iteration()
        else:
            model.fit(X_train_fold, y_train_fold)
        
        # Predict on validation and test data
        oof_preds[val_idx] = model.predict(X_val_fold)
        test_preds += model.predict(X_test)
        
        # Calculate score for this fold
        fold_score = root_mean_squared_error(y_val_fold, oof_preds[val_idx])
        cv_scores[fold] = fold_score
        models.append(model)
        
        print(f"Fold {fold + 1} RMSE: {fold_score:.4f}")
    
    # Summary statistics
    test_preds /= cv_splits
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    best_iteration = best_iterations.mean()
    print(f"Mean CV RMSE: {mean_score:.4f} ± {std_score:.4f}")

    return {\
        "oof_preds": oof_preds,
        "test_preds": test_preds,
        "cv_scores": cv_scores,
        "models": models,
        "best_iteration": int(best_iterations.mean())
    }
