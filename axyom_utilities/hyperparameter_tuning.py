import json
from axyom_utilities.wrappers import XGBRegressorWrapper
from axyom_utilities.training import train_model_cv
import optuna
import torch
from optuna.samplers import TPESampler
from optuna.visualization.matplotlib import (
    plot_optimization_history, 
    plot_param_importances, 
    plot_parallel_coordinate,
    plot_slice,
    plot_contour
)
import matplotlib.pyplot as plt

def tune_xgboost(X_train, y_train, max_time, fixed_params=None, varying_params=None, study_name="xgboost"):
    
    if fixed_params is None: # default fixed values
        fixed_params = {
            "n_estimators": 10000,
            "objective": "reg:squarederror",  # XGBoost regression objective
            "tree_method": "gpu_hist" if torch.cuda.is_available() else "auto",
            "verbosity": 0,
            "enable_categorical": True
        }
    
    if varying_params is None: # default varying ranges
        varying_params = lambda trial: {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 8, 15),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 50, log=True),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "lambda": trial.suggest_float("lambda", 1e-3, 10, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 10, log=True)
        }

    objective = lambda trial: generic_objective(
        trial, 
        XGBRegressorWrapper,
        fixed_params,
        varying_params,
        X_train,
        y_train
    )
    
    best_trial = optuna_tuning(objective, max_time, study_name)
    
    best_params = {**fixed_params, **best_trial.params}

    best_params["n_iterators"] = best_trial.user_attrs.get("best_iteration", None)
    
    with open("xgb_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)


def generic_objective(trial, model_generator, fixed_params, varying_params, X_train, y_train):
    # Define hyperparameter space    

    params = {**fixed_params, **varying_params(trial)}
    
    # Initialize model with trial parameters
    model = model_generator(**params)
    
    # Evaluate using K-Fold CV with early stopping
    results = train_model_cv(\
        model, 
        X_train, 
        y_train, 
        cv_splits=5, 
        early_stopping_rounds=100
    )
    score = results['cv_scores'].mean()

    trial.set_user_attr("best_iteration", results['best_iteration'])
    
    return score

def optuna_tuning(objective, hyper_opt_time, study_name, storage="sqlite:///study.db"):
    # Prepare data
    # Replace X_train, y_train, and X_test with your data
    # Example:
    # X_train, X_test, y_train = ...
    
    # Run Optuna optimization
    study = optuna.create_study( \
        direction="minimize", 
        study_name=study_name, 
        storage=storage, 
        load_if_exists=True,
        sampler=TPESampler(seed=666)
    )
    study.optimize(objective, n_trials=10000, timeout=hyper_opt_time)
    
    # Best parameters and result
    print("Best Trial: ", study.best_trial.params)
    print("Best Score: ", study.best_value)
   
    plot_optimization_history(study)
    plt.show()
    
    plot_param_importances(study)
    plt.show()
    
    plot_slice(study)
    plt.show()

    return study.best_trial


