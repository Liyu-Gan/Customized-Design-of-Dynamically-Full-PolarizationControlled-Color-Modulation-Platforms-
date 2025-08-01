import spectrum as sp
import numpy as np
import numpy as np
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader, TensorDataset, random_split
import spectrum as sp  
from scipy.interpolate import interp1d
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt


# Interpolation function, used to uniformly extract a specified number of data points from the original data
def interpolate_data(data, target_length):
    x_old = np.linspace(0, 1, len(data))
    f = interp1d(x_old, data, kind='linear')
    x_new = np.linspace(0, 1, target_length)
    return f(x_new)

# MinMaxScaler normalization function
def minmaxscaler(data):
    flattened_data = np.array(data).flatten()
    data_list = flattened_data.tolist()
    Min = min(data_list)
    Max = max(data_list)
    unifydata = (data - Min) / (Max - Min)
    outdata = torch.from_numpy(np.array(unifydata).reshape(data.shape))
    return outdata, Min, Max

# MinMaxScaler normalization function
def RGBminmaxscaler(data):
    flattened_data = np.array(data).flatten()
    data_list = flattened_data.tolist()
    Min = 0
    Max = 255
    unifydata = (data - Min) / (Max - Min)
    outdata = torch.from_numpy(np.array(unifydata).reshape(data.shape))
    return outdata, Min, Max

# Data processing functions
def process_data(data_dict, color_data, target_length):
    all_color_features = []
    all_stretch = []
    all_gray = []
    all_a1 = []
    all_a2 = []
    all_a3 = []
    targets = []
    keys = []  # The key name used to store each sample
    # Collect all the data
    for key, value in data_dict.items():
        parts = key.split('_')
        color, stretch, gray, a1, a2, a3 = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])

        if color in color_data:
            color_spectrum = color_data[color]['y']
            color_features = interpolate_data(color_spectrum, target_length)
            all_color_features.append(color_features)
        else:
            continue

        all_stretch.append([stretch])
        all_gray.append([gray])
        all_a1.append([a1])
        all_a2.append([a2])
        all_a3.append([a3])
        target_rgb = [value['r'],value['g'],value['b']]
        targets.append(target_rgb)
        keys.append(key)  # Record the key name corresponding to the current sample

    # Feature Normalization
    norm_color_features, min__color_features, max_color_features = minmaxscaler(np.array(all_color_features))
    norm_stretch, min_stretch, max_stretch = minmaxscaler(np.array(all_stretch))
    norm_gray, min_gray, max_gray = minmaxscaler(np.array(all_gray))
    norm_a1, min_a1, max_a1 = minmaxscaler(np.array(all_a1))
    norm_a2, min_a2, max_a2 = minmaxscaler(np.array(all_a2))
    norm_a3, min_a3, max_a3 = minmaxscaler(np.array(all_a3))
    norm_targets, min_targets, max_targets = RGBminmaxscaler(np.array(targets))

    # Combined feature vector
    features = np.hstack([norm_color_features, norm_stretch, norm_gray, norm_a1, norm_a2, norm_a3])
    return features, np.array(norm_targets), keys

# Custom dataset class
class CustomTensorDataset(Dataset):
    def __init__(self, features, targets, keys):
        self.features = features
        self.targets = targets
        self.keys = keys

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.keys[idx]

    def __len__(self):
        return len(self.features)

# Model definition
class Bias_Batch_Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, out_dim):
        super(Bias_Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.Dropout(p=0.3))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.BatchNorm1d(n_hidden_3), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4), nn.BatchNorm1d(n_hidden_4), nn.Dropout(p=0.3))
        self.output_layer = nn.Linear(n_hidden_4, out_dim)
        self.bias = nn.Parameter(torch.zeros(out_dim))  # Adding an additional bias parameter

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output_layer(x) + self.bias  # Add additional bias to the output
        return x

# Model training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=400):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=600, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        for batch_features, batch_targets, _ in train_loader:
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Evaluating the Model
def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    actuals = []
    predictions = []

    with torch.no_grad():
        for inputs, targets, _ in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            actuals.extend(targets.view(-1).detach().cpu().numpy())
            predictions.extend(outputs.view(-1).detach().cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    print(f'Test Loss: {avg_loss:.4f}')
    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R2 Score: {r2:.4f}')


# Plot the spectrum of the specified sample
def plot_predictions_vs_actuals_for_sample(model, train_loader, test_loader, sample_key):
    model.eval()
    sample_found_in_train = False
    sample_found_in_test = False
    sample_data_loader = None
    sample_data_loader_name = ""
    sample_idx_in_loader = None

    # Find Samples
    with torch.no_grad():
        for i, (inputs, targets, keys) in enumerate(train_loader):
            if sample_key in keys:
                sample_found_in_train = True
                sample_data_loader = train_loader
                sample_data_loader_name = "Training Set"
                sample_idx_in_loader = keys.index(sample_key)
                break

        if not sample_found_in_train:
            for i, (inputs, targets, keys) in enumerate(test_loader):
                if sample_key in keys:
                    sample_found_in_test = True
                    sample_data_loader = test_loader
                    sample_data_loader_name = "Test Set"
                    sample_idx_in_loader = keys.index(sample_key)
                    break

    if not sample_found_in_train and not sample_found_in_test:
        print(f"Sample {sample_key} not found in either Training or Test Set.")
        return

    # Get the predicted and actual values ​​of the sample
    for i, (inputs, targets, keys) in enumerate(sample_data_loader):
        if i == sample_idx_in_loader:
            outputs = model(inputs)
            predicted_spectrum = outputs.view(-1).cpu().numpy()
            actual_spectrum = targets.view(-1).cpu().numpy()

            # Plot the predicted spectrum and the actual spectrum of this sample
            plt.figure(figsize=(10, 6))
            plt.plot(predicted_spectrum, label='Predicted Spectrum', color='blue')
            plt.plot(actual_spectrum, label='Actual Spectrum', color='red', linestyle='--')
            plt.title(f'Predictions vs Actual Spectrum for Sample: {sample_key} ({sample_data_loader_name})')
            plt.xlabel('Wavelength')
            plt.ylabel('Intensity')
            plt.legend()
            plt.grid(True)
            plt.show()

def plot_predictions_vs_actuals(model, data_loader):
    model.eval()
    
    # Initialize a list to store predicted values ​​and true values
    predictions_r, predictions_g, predictions_b = [], [], []
    actuals_r, actuals_g, actuals_b = [], [], []
    
    with torch.no_grad():
        for inputs, targets, _ in data_loader:
            outputs = model(inputs)
            
            # Assume the output is of shape [batch_size, 3], representing R, G, B values
            # Assume the targets are of the same shape [batch_size, 3]
            pred_r, pred_g, pred_b = outputs[:, 0].cpu().numpy(), outputs[:, 1].cpu().numpy(), outputs[:, 2].cpu().numpy()
            true_r, true_g, true_b = targets[:, 0].cpu().numpy(), targets[:, 1].cpu().numpy(), targets[:, 2].cpu().numpy()
            
            # Collect the predicted and true values ​​for each channel
            predictions_r.extend(pred_r)
            predictions_g.extend(pred_g)
            predictions_b.extend(pred_b)
            actuals_r.extend(true_r)
            actuals_g.extend(true_g)
            actuals_b.extend(true_b)

    # Calculate the R² value for each channel
    r2_r = r2_score(actuals_r, predictions_r)
    r2_g = r2_score(actuals_g, predictions_g)
    r2_b = r2_score(actuals_b, predictions_b)

    # Calculate MSE, RMSE, MAE for each channel
    mse_r = mean_squared_error(actuals_r, predictions_r)
    mse_g = mean_squared_error(actuals_g, predictions_g)
    mse_b = mean_squared_error(actuals_b, predictions_b)

    rmse_r = sqrt(mse_r)
    rmse_g = sqrt(mse_g)
    rmse_b = sqrt(mse_b)

    mae_r = mean_absolute_error(actuals_r, predictions_r)
    mae_g = mean_absolute_error(actuals_g, predictions_g)
    mae_b = mean_absolute_error(actuals_b, predictions_b)

    # Print evaluation metrics for each channel
    print(f"R Channel - R²: {r2_r:.4f}, MSE: {mse_r:.4f}, RMSE: {rmse_r:.4f}, MAE: {mae_r:.4f}")
    print(f"G Channel - R²: {r2_g:.4f}, MSE: {mse_g:.4f}, RMSE: {rmse_g:.4f}, MAE: {mae_g:.4f}")
    print(f"B Channel - R²: {r2_b:.4f}, MSE: {mse_b:.4f}, RMSE: {rmse_b:.4f}, MAE: {mae_b:.4f}")

    # Draw a scatter plot of the R channel
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals_r, predictions_r, alpha=0.3, color='red')  
    plt.title(f'R Channel Predictions vs Actuals (R² = {r2_r:.2f}, RMSE = {rmse_r:.2f}, MAE = {mae_r:.2f})')
    plt.xlabel('Actual R Values')
    plt.ylabel('Predicted R Values')
    plt.grid(True)
    plt.show()

    # Draw a scatter plot of the G channel
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals_g, predictions_g, alpha=0.3, color='green') 
    plt.title(f'G Channel Predictions vs Actuals (R² = {r2_g:.2f}, RMSE = {rmse_g:.2f}, MAE = {mae_g:.2f})')
    plt.xlabel('Actual G Values')
    plt.ylabel('Predicted G Values')
    plt.grid(True)
    plt.show()

    # Draw a scatter plot of channel B
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals_b, predictions_b, alpha=0.3, color='blue') 
    plt.title(f'B Channel Predictions vs Actuals (R² = {r2_b:.2f}, RMSE = {rmse_b:.2f}, MAE = {mae_b:.2f})')
    plt.xlabel('Actual B Values')
    plt.ylabel('Predicted B Values')
    plt.grid(True)
    plt.show()


ColorPath=r'..\data_value\color.msgz'
SpecRGBPath = r'..\data_value\spc_rgb_dict.json'
model_save_path = r'..\data_value\RGBmodel.pt'

my_net = Bias_Batch_Net(106, 128, 217, 152, 93, 3)
lr=0.0006 
num_epochs=1000
batch_size=64

'''Reading color data'''
color_data = sp.read_file(ColorPath)
# Reading JSON Files
with open(SpecRGBPath, 'r') as f:
    data_dict = json.load(f)

# Usage examples:
# Perform data processing, model training and evaluation
features, targets, keys = process_data(data_dict, color_data, target_length=101)
features_tensor = torch.tensor(features, dtype=torch.float32)
targets_tensor = torch.tensor(targets, dtype=torch.float32)

dataset = CustomTensorDataset(features_tensor, targets_tensor, keys)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Create model
optimizer = optim.Adam(my_net.parameters(), lr=0.0006, weight_decay=1e-5)
criterion = nn.SmoothL1Loss()

# Train the model
train_model(my_net, train_loader, criterion, optimizer, num_epochs=1000)
# Evaluating the Model
evaluate_model(my_net, train_loader, criterion)
evaluate_model(my_net, test_loader, criterion)

'''Evaluating the prediction performance of three values of R G B'''
# Call the drawing function, passing in the model and data loader
plot_predictions_vs_actuals(my_net, test_loader)

'''Save the model'''
torch.save(my_net.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')


