import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a simple neural network model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        return x

def train_model(input_file_path, output_file_path):
    # Load input data from the JSON file
    with open(input_file_path, 'r') as input_file:
        input_data = json.load(input_file)
    
    # Extract features and labels from the input data
    features = np.array(input_data['features'])
    labels = np.array(input_data['labels'])

    # Convert data to PyTorch tensors
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    # Initialize the model, loss function, and optimizer
    model = SimpleModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    for epoch in range(1000):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    # Save the trained model as a PyTorch file
    torch.save(model.state_dict(), output_file_path)

if __name__ == "__main__":
    # Path to the input and output data JSON files
    input_file_path = '/input_data/input.json'
    output_file_path = '/output_data/model.pt'
    
    # Train the model using input data and save the trained model
    train_model(input_file_path, output_file_path)

import json
import pickle
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def train_model(input_file_path, output_file_path):
    # Load input data from the JSON file
    with open(input_file_path, 'r') as input_file:
        input_data = json.load(input_file)
    
    # Extract features and labels from the input data
    features = np.array(input_data['features'])
    labels = np.array(input_data['labels'])

    # Scale or normalize the input features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Initialize the model with adjusted hyperparameters
    model = MLPRegressor(hidden_layer_sizes=(2,), activation='logistic', max_iter=1000, learning_rate_init=0.01, alpha=0.001, random_state=42)

    # Train the model
    model.fit(features_scaled, labels)

    # Save the trained model using pickle
    with open(output_file_path, 'wb') as output_file:
        pickle.dump((model, scaler), output_file)

if __name__ == "__main__":
    # Path to the input and output data JSON files
    input_file_path = '/input_file/input.json'
    output_file_path = '/output_data/model.pkl'
    
    # Train the model using input data and save the trained model
    train_model(input_file_path, output_file_path)
# print("Hello")

# while True:
#     pass