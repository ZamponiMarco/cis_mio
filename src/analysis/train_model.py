import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def train_and_save_model(data_file, model_type, output_file, parameters, max_depth, n_estimators=None):
    """
    Loads dataset, trains the specified model, and saves it.

    Args:
        data_file: path to input data file
        model_type: type of model to train ('decision_tree', 'random_forest')
        output_file: path to output file containing the trained model
        parameters: list of parameters acting as input data
        max_depth: maximum depth of the tree
        n_estimators: number of trees in the tree ('random_forest' only)
    """

    data = pd.read_csv(data_file)
    data = data[data['delta_0'] != -1]

    X = data[parameters]
    y = data.drop(parameters, axis=1)

    if model_type == 'random_forest':
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    elif model_type == 'decision_tree':
        clf = DecisionTreeClassifier(max_depth=max_depth)
    else:
        raise ValueError("Invalid model type. Choose 'random_forest' or 'decision_tree'.")

    clf.fit(X, y)

    joblib.dump(clf, output_file)
    print(f"Trained and saved {model_type.replace('_', ' ').title()} Classifier to {output_file}.")

    os.remove(data_file)
