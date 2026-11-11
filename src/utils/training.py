import numpy as np
import xgboost as xgb
from utils.calculate_ams_score import calculate_ams_score

def objective(trial, X_train, y_train, X_val, y_val, w_val, use_gpu=False):
    """Optuna objective function to maximize AMS score"""
    param = {
        # These set guidelines for hyperparameter tuning and gives a range to explore
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5)
    }

    # Dynamic variable so that the pipeline can run on either MacOS (CPU) or OpenShift x64 (GPUs)
    if use_gpu:
        param['tree_method'] = 'hist'
        param['device'] = 'cuda'

    # Train the XGBoost model with the hyperparameter guidelines
    clf = xgb.XGBClassifier(**param)

    # Fit the model on the training data
    clf.fit(X_train, y_train)

    # Predict probabilities for the validation set
    y_pred_proba = clf.predict_proba(X_val)[:, 1]

    # Evaluate the model using the AMS score
    best_ams_for_trial = 0

    # Use dynamic probability thresholds to find the best AMS score for this trial
    # Thresholds range from 0.70 to 0.95 with 50 values in between
    for threshold in np.linspace(0.70, 0.95, 50): 
        y_pred = (y_pred_proba >= threshold).astype(int)
        current_ams = calculate_ams_score(y_val, y_pred, w_val, data_split_ratio=0.2) 
        if current_ams > best_ams_for_trial:
            best_ams_for_trial = current_ams

    # Return the best AMS score for this trial to Optuna for optimisation
    return best_ams_for_trial 
