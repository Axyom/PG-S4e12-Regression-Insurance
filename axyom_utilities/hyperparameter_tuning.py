import json
from axyom_utilities.wrappers import XGBRegressorWrapper, LGBMRegressorWrapper
from axyom_utilities.training import train_model_cv
import optuna
import torch
from optuna.samplers import TPESampler
from optuna.visualization.matplotlib import (
    plot_optimization_history, 
    plot_param_importances, 
    plot_slice
)
import matplotlib.pyplot as plt

class ModelTuner:
    def __init__(self, X_train, y_train, max_time, study_name, fixed_params=None, varying_params=None):
        self.X_train = X_train
        self.y_train = y_train
        self.max_time = max_time
        self.study_name = study_name
        self.fixed_params = fixed_params or {}
        self.varying_params = varying_params or (lambda trial: {})

    def tune(self, model_generator, params_file):
        objective = lambda trial: self.generic_objective(trial, model_generator)
        best_trial = self.optuna_tuning(objective)
        
        best_params = {**self.fixed_params, **best_trial.params}
        best_params["n_estimators"] = best_trial.user_attrs.get("best_iteration", None)

        with open(params_file, "w") as f:
            json.dump(best_params, f, indent=4)

        return best_params

    def generic_objective(self, trial, model_generator):
        params = {**self.fixed_params, **self.varying_params(trial)}
        model = model_generator(**params)
        
        results = train_model_cv(
            model, 
            self.X_train, 
            self.y_train, 
            cv_splits=5, 
            early_stopping_rounds=100
        )
        
        score = results['cv_scores'].mean()
        trial.set_user_attr("best_iteration", results['best_iteration'])
        return score

    def optuna_tuning(self, objective):
        study = optuna.create_study(
            direction="minimize", 
            study_name=self.study_name, 
            storage="sqlite:///study.db", 
            load_if_exists=True,
            sampler=TPESampler(seed=666)
        )
        study.optimize(objective, n_trials=10000, timeout=self.max_time)

        print("Best Trial: ", study.best_trial.params)
        print("Best Score: ", study.best_value)

        plot_optimization_history(study)
        plt.show()

        plot_param_importances(study)
        plt.show()

        plot_slice(study)
        plt.show()

        return study.best_trial

class LGBMTuner(ModelTuner):
    def __init__(self, X_train, y_train, max_time, study_name="lgbm", fixed_params=None, varying_params=None):
        default_fixed_params = {
            "n_estimators": 10000,
            "objective": "regression",
            "metric": "rmse",  # Use RMSE as the evaluation metric
            "device": "gpu",
            "verbose": -1
        }
        fixed_params = {**default_fixed_params, **(fixed_params or {})}
        
        default_varying_params = lambda trial: {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 8, 15),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 50, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10, log=True)
        }
        varying_params = varying_params or default_varying_params
        
        super().__init__(X_train, y_train, max_time, study_name, fixed_params, varying_params)

    def tune(self):
        return super().tune(LGBMRegressorWrapper, "lgbm_best_params.json")

class XGBoostTuner(ModelTuner):
    def __init__(self, X_train, y_train, max_time, study_name="xgboost", fixed_params=None, varying_params=None):
        default_fixed_params = {
            "n_estimators": 10000,
            "objective": "reg:squarederror",
            "tree_method": "gpu_hist" if torch.cuda.is_available() else "auto",
            "verbosity": 0,
            "enable_categorical": True
        }
        fixed_params = {**default_fixed_params, **(fixed_params or {})}
        
        default_varying_params = lambda trial: {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 8, 15),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 50, log=True),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "lambda": trial.suggest_float("lambda", 1e-3, 10, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 10, log=True)
        }
        varying_params = varying_params or default_varying_params
        
        super().__init__(X_train, y_train, max_time, study_name, fixed_params, varying_params)

    def tune(self):
        return super().tune(XGBRegressorWrapper, "xgb_best_params.json")
