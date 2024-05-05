import json
import pickle
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import logging

logging.basicConfig(level=logging.INFO)

def train_model(input_file_path, output_file_path):
    logging.info(f"Loading data from {input_file_path}")
    with open(input_file_path, 'r') as input_file:
        input_data = json.load(input_file)

    features = np.array(input_data['features'])
    labels = np.array(input_data['labels'])

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    model = MLPRegressor(hidden_layer_sizes=(2,), activation='logistic', max_iter=1000,
                         learning_rate_init=0.01, alpha=0.001, random_state=42)
    model.fit(features_scaled, labels)

    logging.info(f"Saving model to {output_file_path}")
    with open(output_file_path, 'wb') as output_file:
        pickle.dump((model, scaler), output_file)
    logging.info("Model training complete and file saved.")

if __name__ == "__main__":
    input_file_path = '/shared_data/input.json'
    output_file_path = '/shared_data/model.pkl'

    train_model(input_file_path, output_file_path)
