from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd
from xgboost.callback import EarlyStopping

class XGBRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.params = kwargs

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, verbose=False):
        """
        Train the XGBRegressor model.

        Parameters:
        - X: pd.DataFrame or array-like
          Training features.
        - y: array-like
          Training labels.
        - eval_set: tuple or None
          Optional validation set for early stopping, in the form [(X_val, y_val)].
        - early_stopping_rounds: int or None
          Number of rounds for early stopping. Set to None to disable.
        - verbose: bool
          Whether to print training progress.
        """
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Initialize and train the XGB Regressor
        self.params["early_stopping_rounds"] = early_stopping_rounds 
        
        self.xgb_model_ = XGBRegressor(**self.params)
        
        # callbacks = []
        # if early_stopping_rounds and eval_set:
        #     callbacks.append(EarlyStopping(rounds=early_stopping_rounds, save_best=True, maximize=False, metric_name="rmse"))


        # Train the model with early stopping if validation set is provided
        self.xgb_model_.fit(
            X,
            y,
            eval_set=[eval_set],
            #early_stopping_rounds=early_stopping_rounds,
            #callbacks=callbacks,
            verbose=verbose
        )
        
        return self

    def predict(self, X):
        return self.xgb_model_.predict(X)

    def get_best_iteration(self):
        """
        Get the best iteration for early stopping.
        """
        return self.xgb_model_.best_iteration if hasattr(self.xgb_model_, "best_iteration") else None

    def get_params(self, deep=True):
        return self.params
    
    def set_params(self, **parameters):
        self.params.update(parameters)
        return self
