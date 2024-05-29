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
# import pickle
# import numpy as np
# import math
# import json
# import pandas as pd
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
# from sklearn.model_selection import GridSearchCV
# from skopt import gp_minimize
# from skopt.space import Real
# from sklearn.metrics import mean_squared_error
# import os

# def sensors_value_to_matrix(dict_sensors_t):
#     X_rs = []
#     Y_rs = []
#     X_tot = []
#     Y_tot = []
#     Z_rs = []
#     for sensor in dict_sensors_t.keys():
#         if not (np.isnan(dict_sensors_t[sensor]["value"]) | math.isnan(dict_sensors_t[sensor]["value"]) | pd.isna(dict_sensors_t[sensor]["value"])):
#             X_rs.append(dict_sensors_t[sensor]['x'])
#             Y_rs.append(dict_sensors_t[sensor]['y'])
#             Z_rs.append(dict_sensors_t[sensor]['value'])
#         X_tot.append(dict_sensors_t[sensor]['x'])
#         Y_tot.append(dict_sensors_t[sensor]['y'])
#     X_rs = np.array(X_rs)
#     X_tot = np.array(X_tot)
#     Y_rs = np.array(Y_rs)
#     Y_tot = np.array(Y_tot)
#     Z_rs = np.array(Z_rs)
    
#     y_mean, y_std = np.mean(Y_rs), np.std(Y_rs)
#     Y_tot = (Y_tot - y_mean) / y_std
#     Y_rs = (Y_rs - y_mean) / y_std

#     x_mean, x_std = np.mean(X_rs), np.std(X_rs)
#     X_tot = (X_tot - x_mean) / x_std
#     X_rs = (X_rs - x_mean) / x_std

#     z_mean = np.mean(Z_rs)
#     z_std = np.std(Z_rs)
#     Z_rs = (Z_rs - z_mean) / z_std

#     return {'x': X_rs, 'y': Y_rs, 'x_tot': X_tot, 'y_tot': Y_tot, 'z': Z_rs, 'z_mean': z_mean, 'z_std': z_std, 'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std}

# def all_training_values(dict_sensors):
#     all_train = []
#     for timestamp in dict_sensors.keys():
#         dict_matrix_values = sensors_value_to_matrix(dict_sensors[timestamp])
#         A_train = np.column_stack((dict_matrix_values['x'], dict_matrix_values['y']))
#         Z_train = dict_matrix_values['z']
#         all_train.append((A_train, Z_train))
#     return all_train

# def hyperparameters_training(dict_sensors, bounds_constant_value=np.logspace(-1, 1, 10), bounds_length_scale=np.logspace(-1, 1, 10), kernel=None, model=None):
#     all_train = all_training_values(dict_sensors)
#     param_space = [Real(bounds_constant_value[0], bounds_constant_value[1], name='k1'),
#                    Real(bounds_length_scale[0], bounds_length_scale[1], name='k2')]
#     if kernel is None:
#         kernel = matern_kernel(bounds_constant_value[0], bounds_length_scale[0], 1/2)
#     if model is None:
#         model = GaussianProcessRegressor(kernel=kernel)
    
#     def objective(params):
#         k1, k2 = params
#         mse_total = 0
#         for (A_train, Z_train) in all_train:
#             model.kernel.k1.constant_value = k1
#             model.kernel.k2.length_scale = k2
#             model.fit(A_train, Z_train)
#             Z_pred = model.predict(A_train)
#             mse_total += mean_squared_error(Z_train, Z_pred)
#         return mse_total / len(all_train)

#     param_space = [Real(bounds_constant_value[0], bounds_constant_value[-1], name='k1'),
#                    Real(bounds_length_scale[0], bounds_length_scale[-1], name='k2')]
#     result = gp_minimize(objective, param_space, n_calls=20, random_state=42)
#     best_k1, best_k2 = result.x
#     kernel = matern_kernel(best_k1, best_k2, 1/2)
#     model = GaussianProcessRegressor(kernel=kernel)
#     return model

# def matern_kernel(constant_value, length_scale, nu):
#     return ConstantKernel(constant_value=constant_value) * Matern(length_scale=length_scale, nu=nu)

# class GPRModel():
#     def __init__(self, constant_value=1, length_scale=1, nu=3/2):
#         self.constant_value = constant_value
#         self.length_scale = length_scale
#         self.kernel = ConstantKernel(constant_value=constant_value) * Matern(length_scale=length_scale, nu=nu)
#         self.model = GaussianProcessRegressor(kernel=self.kernel)

#     def training(self, dict_sensors):
#         self.model = hyperparameters_training(dict_sensors, bounds_constant_value=[1e-1, 1e1], bounds_length_scale=[1e-1, 1e1], kernel=self.kernel, model=self.model)
#         self.kernel = self.model.kernel
#         self.constant_value = self.kernel.k1.constant_value
#         self.length_scale = self.kernel.k2.length_scale
    
#     def predict(self, snapshot):
#         mod = self.model
#         dict_matrix_val = sensors_value_to_matrix(snapshot)
#         A_fit = np.column_stack((dict_matrix_val['x'], dict_matrix_val['y']))
#         Z_fit = dict_matrix_val['z']
#         A = np.column_stack((dict_matrix_val['x_tot'], dict_matrix_val['y_tot']))
#         mod = mod.fit(A_fit, Z_fit)
#         return A, mod.predict(A)

#     def reset_parameters(self):
#         self.constant_value = 1
#         self.length_scale = 1
#         self.kernel = ConstantKernel(constant_value=self.constant_value) * Matern(length_scale=self.length_scale, nu=3/2)
#         self.model = GaussianProcessRegressor(kernel=self.kernel)

#     def save_model(self, save_dir):
#         with open(save_dir, 'wb') as file:
#             pickle.dump(self, file)

# if __name__ == "__main__":
#     # Define file paths
#     input_file_path = '/shared_data/input.json'  # Input data file
#     output_file_path = '/shared_data/model.pkl'  # Output model file


#     # Train the model and save
#     with open(input_file_path, 'r') as input_file:
#         input_data = json.load(input_file)
    
#     model = GPRModel()
#     model.training(input_data)
#     model.save_model(output_file_path)

#     # Verify if the model file is created
#     if os.path.exists(output_file_path):
#         print(f"Model saved successfully at {output_file_path}")
#     else:
#         print("Model saving failed.")
