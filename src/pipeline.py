import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def load_preprocessed_data():
    print("Loading dataset...")

    # Load the dataset from the 2014 Challenge CSV file
    df = pd.read_csv('data/atlas-higgs-challenge-2014-v2.csv')

    # The CERN dataset uses -999 to signify missing values.
    # Replaced with Not Available for better handling in pandas.
    df.replace(-999.0, pd.NA, inplace=True)

    # Drop the columns that are not used to predict the label.
    X = df.drop(columns=["EventId", "Weight", "Label"], axis=1)

    # Encode the labels as integers for the model. Binary classification: 0 for background events, 1 for signal events.
    # This is used to distinguish between the two classes for the prediction algorithm.
    encoder = LabelEncoder()

    # Fit the encoder to the labels and transform them into integer values.
    y = encoder.fit_transform(df["Label"])

    # Split the dataset into training and testing sets. 80% of the data is used for training, and 20% is used for testing.
    return train_test_split(X, y, test_size=0.2, random_state=42)