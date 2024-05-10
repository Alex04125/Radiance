import json
import pickle
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def train_model(input_file_path, output_file_path):
    # Load input data from the JSON file
    with open(input_file_path, 'r') as input_file:
        input_data = json.load(input_file)
    
    # Extract features and labels
    features = np.array(input_data['features'])
    labels = np.array(input_data['labels'])

    # Initialize and fit the scaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Initialize and train the model
    model = MLPRegressor(
        hidden_layer_sizes=(2,),
        activation='logistic',
        max_iter=1000,
        learning_rate_init=0.01,
        alpha=0.001,
        random_state=42
    )
    model.fit(features_scaled, labels)

    # Save only the model to the output pickle file
    with open(output_file_path, 'wb') as output_file:
        pickle.dump(model, output_file)

if __name__ == "__main__":
    # Define file paths
    input_file_path = '/shared_data/input.json'  # Input data file
    output_file_path = '/shared_data/model.pkl'  # Output model file

    # Train the model and save
    train_model(input_file_path, output_file_path)
