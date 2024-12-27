from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd
from catboost import CatBoostRegressor, Pool

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd

import ydf  # Yggdrasil Decision Forests
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd

class YggdrasilRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.params = kwargs

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, verbose=False):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        train_data = X.copy()
        train_data['target'] = y
        self.yggdrasil_model_ = ydf.GradientBoostedTreesLearner(
            label="target",
            early_stopping="VALIDATION_LOSS_INCREASE" if early_stopping_rounds else "NONE",
            early_stopping_num_trees_look_ahead=early_stopping_rounds if early_stopping_rounds else 0,
            **self.params
        )
        self.yggdrasil_model_.train(train_data, verbose=verbose)
        return self

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return self.yggdrasil_model_.predict(X)

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **parameters):
        self.params.update(parameters)
        return self


class HGBMRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        """
        Parameters:
        - kwargs: Additional parameters for HistGradientBoostingRegressor.
        """
        self.params = kwargs

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, verbose=False):
        """
        Train the HistGradientBoostingRegressor model.

        Parameters:
        - X: pd.DataFrame or array-like
          Training features.
        - y: array-like
          Training labels.
        - eval_set: tuple or None
          Optional validation set for early stopping, in the form (X_val, y_val).
        - early_stopping_rounds: int or None
          Number of rounds for early stopping. Set to None to disable.
        - verbose: bool
          Whether to print training progress.
        """
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Prepare validation data
        self.params['early_stopping'] = early_stopping_rounds is not None
        self.params['validation_fraction'] = 0.1 if eval_set else None
        self.params['n_iter_no_change'] = early_stopping_rounds

        if eval_set:
            X_val, y_val = eval_set
            self.params['validation_fraction'] = len(y_val) / (len(y) + len(y_val))

        # Initialize and train the HGBM Regressor
        self.hgbm_model_ = HistGradientBoostingRegressor(**self.params)
        self.hgbm_model_.fit(X, y)

        return self

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        - X: pd.DataFrame or array-like
          Features to predict on.

        Returns:
        - array-like: Predicted values.
        """
        return self.hgbm_model_.predict(X)

    def get_best_iteration(self):
        """
        HistGradientBoostingRegressor does not directly provide best iteration.
        """
        return None

    def get_params(self, deep=True):
        """
        Get parameters of the HGBM model.

        Returns:
        - dict: Model parameters.
        """
        return self.params

    def set_params(self, **parameters):
        """
        Set parameters for the HGBM model.

        Parameters:
        - parameters: dict
          Parameters to update.

        Returns:
        - self
        """
        self.params.update(parameters)
        return self


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

class CatBoostRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        """
        Parameters:
        - kwargs: Additional parameters for CatBoostRegressor.
        """
        self.params = kwargs

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, verbose=False):
        """
        Train the CatBoostRegressor model.

        Parameters:
        - X: pd.DataFrame
          Training features.
        - y: array-like
          Training labels.
        - eval_set: tuple or None
          Optional validation set for early stopping, in the form (X_val, y_val).
        - early_stopping_rounds: int or None
          Number of rounds for early stopping. Set to None to disable.
        - verbose: bool
          Whether to print training progress.
        """
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Identify categorical columns
        categorical_columns = X.select_dtypes(include=['object', 'category', 'string']).columns
        categorical_indices = [X.columns.get_loc(col) for col in categorical_columns]

        # Create CatBoost Pool with categorical features
        training_pool = Pool(X, y, cat_features=categorical_indices)
        eval_pool = None
        if eval_set is not None:
            X_val, y_val = eval_set
            eval_pool = Pool(X_val, y_val, cat_features=categorical_indices)

        # Initialize and train the CatBoost Regressor
        self.catboost_model_ = CatBoostRegressor(**self.params)
        self.catboost_model_.fit(
            training_pool,
            eval_set=eval_pool,
            early_stopping_rounds=early_stopping_rounds
        )
        
        return self

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        - X: pd.DataFrame or array-like
          Features to predict on.

        Returns:
        - array-like: Predicted values.
        """
        return self.catboost_model_.predict(X)

    def get_best_iteration(self):
        return self.catboost_model_.get_best_iteration()

    def get_params(self, deep=True):
        """
        Get parameters of the CatBoost model.

        Returns:
        - dict: Model parameters.
        """
        return self.params
    
    def set_params(self, **parameters):
        """
        Set parameters for the CatBoost model.

        Parameters:
        - parameters: dict
          Parameters to update.

        Returns:
        - self
        """
        self.params.update(parameters)
        return self
      
from lightgbm import LGBMRegressor
from lightgbm import early_stopping

class LGBMRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.params = kwargs

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, verbose=False):
        """
        Train the LGBMRegressor model.

        Parameters:
        - X: pd.DataFrame or array-like
          Training features.
        - y: array-like
          Training labels.
        - eval_set: tuple or None
          Optional validation set for early stopping, in the form (X_val, y_val).
        - early_stopping_rounds: int or None
          Number of rounds for early stopping. Set to None to disable.
        - verbose: bool
          Whether to print training progress.
        """
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Initialize and train the LGBM Regressor
        self.lgbm_model_ = LGBMRegressor(**self.params)

        # Prepare callbacks for early stopping and logging
        callbacks = []
        if early_stopping_rounds:
            callbacks.append(early_stopping(early_stopping_rounds))
        # if verbose:
        #     callbacks.append(log_evaluation(period=1))
        
        self.lgbm_model_.fit(
            X,
            y,
            eval_set=eval_set,
            callbacks=callbacks
        )
        
        return self

    def predict(self, X):
        return self.lgbm_model_.predict(X)

    def get_best_iteration(self):
        return self.lgbm_model_.best_iteration_ if hasattr(self.lgbm_model_, "best_iteration_") else None

    def get_params(self, deep=True):
        return self.params
    
    def set_params(self, **parameters):
        self.params.update(parameters)
        return self