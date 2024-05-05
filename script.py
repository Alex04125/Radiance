import json
import pickle
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def train_model(input_file_path, output_file_path):
    with open(input_file_path, 'r') as input_file:
        input_data = json.load(input_file)

    features = np.array(input_data['features'])
    labels = np.array(input_data['labels'])

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    model = MLPRegressor(hidden_layer_sizes=(2,), activation='logistic', max_iter=1000,
                         learning_rate_init=0.01, alpha=0.001, random_state=42)
    model.fit(features_scaled, labels)

    with open(output_file_path, 'wb') as output_file:
        pickle.dump((model, scaler), output_file)

if __name__ == "__main__":
    input_file_path = '/shared_data/input.json'  # Input data file
    output_file_path = '/shared_data/model.pkl'  # Output model file

    train_model(input_file_path, output_file_path)
