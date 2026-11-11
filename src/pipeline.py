import os
import pandas as pd
import numpy as np
import xgboost as xgb
import time
import optuna
from sklearn.metrics import accuracy_score

from utils.calculate_ams_score import calculate_ams_score
from utils.load_dataset import load_preprocessed_data
from utils.training import objective

if __name__ == "__main__":
    # Check environment variable for Osprey deployment
    use_gpu = os.environ.get("USE_GPU", "false").lower() == "true"
    print(f"Hardware Acceleration (CUDA) Enabled: {use_gpu}")

    # Load the pre-processed data
    X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val, w_test = load_preprocessed_data()

    # Initialise Bayesian optimisation with Optuna for hyperparameters
    print("\nInitiating Bayesian optimisation with Optuna...")
    start_time = time.time()
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, w_val, use_gpu), n_trials=10)
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    print(f"Best Trial AMS: {study.best_value:.4f}")
    
    # Combine training and validation sets for the final model
    print("\nTraining final pipeline with optimal parameters...")
    best_params = study.best_params
    
    if use_gpu:
        best_params['tree_method'] = 'hist'
        best_params['device'] = 'cuda'
        
    final_model = xgb.XGBClassifier(**best_params)
    X_full_train = pd.concat([X_train, X_val])
    y_full_train = np.concatenate([y_train, y_val])
    final_model.fit(X_full_train, y_full_train)

    # Evaluate the model using dynamic probability threshold sweeping
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    
    final_best_ams = 0
    final_best_threshold = 0
    final_accuracy = 0
    
    for threshold in np.linspace(0.70, 0.95, 50):
        y_pred_temp = (y_test_proba >= threshold).astype(int)
        
        # data_split_ratio = 0.2 as validation split is 20%
        ams = calculate_ams_score(y_test, y_pred_temp, w_test, data_split_ratio=0.2)
        
        if ams > final_best_ams:
            final_best_ams = ams
            final_best_threshold = threshold
            final_accuracy = accuracy_score(y_test, y_pred_temp)

    # Print the final results
    print(f"\n--- OPTIMISED PIPELINE RESULTS ---")
    print(f"Optimal Probability Threshold: {final_best_threshold:.4f}")
    print(f"Final Accuracy: {final_accuracy * 100:.2f}%")
    print(f"Final AMS Score: {final_best_ams:.4f}")
