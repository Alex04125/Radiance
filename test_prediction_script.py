import json
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def load_input_data(input_json_path):
    with open(input_json_path, 'r') as file:
        input_data = json.load(file)
    return np.array(input_data['features'])

def save_predictions(predictions, output_json_path):
    with open(output_json_path, 'w') as file:
        json.dump({'predictions': predictions.tolist()}, file)

def infer(model_path, input_json_path, output_json_path):
    # Load the model
    model = load_model(model_path)
    
    # Load the input data
    features = load_input_data(input_json_path)
    
    # Apply the same scaling
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)  # Assuming same scaling as training
    
    # Make predictions
    predictions = model.predict(features_scaled)
    
    # Save the predictions
    save_predictions(predictions, output_json_path)

if __name__ == "__main__":
    # Define file paths
    MODEL_PATH = '/app/model.pkl'  # Path to the saved model
    INPUT_JSON_PATH = '/shared_data/vs/input.json'  # Path to the input JSON file
    OUTPUT_JSON_PATH = '/shared_data/vs/output.json'  # Path to the output JSON file

    # Run inference and save predictions
    infer(MODEL_PATH, INPUT_JSON_PATH, OUTPUT_JSON_PATH)
