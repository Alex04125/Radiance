# import json
# import os

# def calculate_sum_from_file(input_file_path):
#     # Read input data from the JSON file
#     with open(input_file_path, 'r') as input_file:
#         input_data = json.load(input_file)
    
#     # Extract values of 'a' and 'b' from the input data
#     a = input_data['a']
#     b = input_data['b']
    
#     # Calculate the sum of 'a' and 'b'
#     result = calculate_sum(a, b)
#     return result

# def calculate_sum(a, b):
#     """
#     Calculate the sum of two numbers.

#     Parameters:
#         a (float): The first number.
#         b (float): The second number.

#     Returns:
#         float: The sum of a and b.
#     """
#     return a + b

# if __name__ == "__main__":
#     # Path to the input data JSON file
#     input_file_path = '/input_file/input.json'
    
#     # Calculate the sum using input data from the JSON file
#     result = calculate_sum_from_file(input_file_path)
    
#     # Path to the output file where the result will be saved
#     output_file_path = '/output_data/output.json'
    
#     # Write the result to the output file
#     with open(output_file_path, 'w') as output_file:
#         json.dump({"result": result}, output_file)

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
