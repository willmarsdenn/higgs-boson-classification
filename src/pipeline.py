import numpy as np
import xgboost as xgb
import time
from sklearn.metrics import accuracy_score

from utils.calculate_ams_score import calculate_ams_score
from utils.load_dataset import load_preprocessed_data

if __name__ == "__main__":
    # Load the pre-processed data
    X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val, w_test = load_preprocessed_data()

    # Initialise XGBoost classifier with baseline hyperparameters
    baseline_params = {
        'n_estimators': 150,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    }
    
    start_time = time.time()
    
    model = xgb.XGBClassifier(**baseline_params)
    model.fit(X_train, y_train)
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # Evaluate the model using a standard 50% probability threshold
    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_pred_naive = (y_val_proba >= 0.50).astype(int)
    
    accuracy = accuracy_score(y_val, y_pred_naive)
    
    # data_split_ratio = 0.2 as validation split is 20%
    ams_score = calculate_ams_score(y_val, y_pred_naive, w_val, data_split_ratio=0.2)

    print(f"Baseline Accuracy: {accuracy * 100:.2f}%")
    print(f"Baseline AMS Score: {ams_score:.4f}")
