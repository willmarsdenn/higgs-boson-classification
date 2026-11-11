import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_preprocessed_data():
    """Loads and pre-processes the CERN ATLAS dataset."""

    # Load the raw CSV dataset
    df = pd.read_csv('./data/atlas-higgs-challenge-2014-v2.csv')

    # Convert classification labels from strings ('s', 'b') to binary integers (1, 0)
    df['Label'] = df['Label'].map({'s': 1, 'b': 0})
    
    # Replace CERN's missing values (-999.0) with standard Not-a-Number (NaN)
    df = df.replace(-999.0, np.nan)
    
    # Extract target variable (y), weights, and features (X)
    y = df['Label'].values
    weights = df['Weight'].values
    
    # Drop non-feature columns (EventId is just an index, Label is target, Weight is for AMS)
    X = df.drop(columns=['EventId', 'Label', 'Weight', 'KaggleWeight', 'KaggleSet'])
    
    # Change all remaining numbers to float64 for compatibility
    X = X.astype('float64')

    # Split the data
    # 20% allocated to testing
    X_temp, X_test, y_temp, y_test, w_temp, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42, stratify=y
    )
    
    # 60% of the remaining 80% is allocated to training, 20% to validation
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_temp, y_temp, w_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    # Print the number of rows in each split for verification
    print(f"Training rows: {len(X_train)} | Validation rows: {len(X_val)} | Test rows: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val, w_test
