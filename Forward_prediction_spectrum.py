import spectrum as sp
import numpy as np
import torch
import json
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import spectrum as sp  # Assuming there is already a library suitable for processing
from scipy.interpolate import interp1d
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Interpolation function, used to evenly extract 121 data points from the original data
def interpolate_data(data, target_length):
    x_old = np.linspace(0, 1, len(data))
    f = interp1d(x_old, data, kind='linear')
    x_new = np.linspace(0, 1, target_length)
    return f(x_new)
def minmaxscaler(data):   # MinMaxScaler normalization method
    # First expand it into a one-dimensional array
    flattened_data = np.array(data).flatten() 
    data_list=flattened_data.tolist()
    Min = min(data_list)
    Max = max(data_list)    
    unifydata=(data - Min)/(Max-Min)
    outdata=torch.from_numpy(np.array(unifydata).reshape(data.shape))
    return outdata,Min,Max

def process_data(data_dict, color_data,target_length):
    all_color_features = []
    all_stretch = []
    all_gray = []
    all_a1 = []
    all_a2 = []
    all_a3 = []
    targets = []
    keys = []  # The key name used to store each sample
    # First collect all the data
    for key, value in data_dict.items():
        parts = key.split('_')
        color, stretch, gray, a1, a2, a3 = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        
        if color in color_data:
            color_spectrum = color_data[color]['y']
            color_features = interpolate_data(color_spectrum,target_length)
            all_color_features.append(color_features)
        else:
            continue

        all_stretch.append([stretch])
        all_gray.append([gray])
        all_a1.append([a1])
        all_a2.append([a2])
        all_a3.append([a3])
        
        target_spectrum = interpolate_data(value['y'],target_length)
        targets.append(target_spectrum)
        keys.append(key)  # Record the key name corresponding to the current sample

    # Then normalize all features
    norm_color_features, min__color_features, max_color_features = minmaxscaler(np.array(all_color_features))
    norm_stretch, min_stretch, max_stretch = minmaxscaler(np.array(all_stretch))
    norm_gray, min_gray, max_gray = minmaxscaler(np.array(all_gray))
    norm_a1, min_a1, max_a1 = minmaxscaler(np.array(all_a1))
    norm_a2, min_a2, max_a2 = minmaxscaler(np.array(all_a2))
    norm_a3, min_a3, max_a3= minmaxscaler(np.array(all_a3))
    norm_targets, min_targets, max_targets = minmaxscaler(np.array(targets))

    # Combined feature vector
    features = np.hstack([norm_color_features, norm_stretch, norm_gray, norm_a1, norm_a2, norm_a3])
    return features, np.array(norm_targets), keys, all_color_features, norm_color_features, min__color_features, max_color_features, min_stretch, max_stretch, min_gray, max_gray, min_a1, max_a1, min_a2, max_a2, min_a3, max_a3, min_targets, max_targets


def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    actuals = []
    predictions = []

    with torch.no_grad():
        for inputs, targets in data_loader:
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

def plot_predictions_vs_actuals(model, data_loader):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            # Store the predicted values ​​and the true values
            predictions.extend(outputs.view(-1).cpu().numpy())
            actuals.extend(targets.view(-1).cpu().numpy())

    # Draw a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.3, color='blue')  # Alpha is used to adjust the transparency of the point
    plt.title('Predictions vs. Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    plt.grid(True)
    plt.show()

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

'''File storage path'''
ColorPath=r'..\data_value\color.msgz'
ExperdataPath = r'..\data_value\all_data.json'
model_save_path = os.path.join(r'..\data_value\zzbmodel.pth')
'''Hyperparameter settings'''
seed_num = 42
scale = 0.8
batch_size = 76
learning_rate = 0.001
weight_decay=1e-5
num_epochs = 400
step_size=400
gamma=0.1
my_net = Bias_Batch_Net(106, 226, 394, 372, 220, 101)

'''Read color spectrum data'''
color_data = sp.read_file(ColorPath)

'''Read experimental spectrum file'''
with open(ExperdataPath, 'r', encoding='utf-8') as f:
    data_dict = json.load(f)

'''Data preprocessing: processing feature and spectral data, including interpolation, normalization, conversion to tensor, etc.'''
features, targets,keys, all_color_features,norm_color_features, min__color_features, max_color_features, min_stretch, max_stretch, min_gray, max_gray, min_a1, max_a1, min_a2, max_a2, min_a3, max_a3, min_targets, max_targets= process_data(data_dict, color_data,101)
features_tensor = torch.tensor(features, dtype=torch.float32)
targets_tensor = torch.tensor(targets, dtype=torch.float32)

'''Divide the dataset'''
# Set the random number seed
torch.manual_seed(seed_num)
# Creating a TensorDataset
dataset = TensorDataset(features_tensor, targets_tensor)
# Divide the data into training and testing sets
train_size = int(scale * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
# Creating a DataLoader
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
# define some hyperparameters
optimizer = optim.Adam(my_net.parameters(), lr=learning_rate, weight_decay =weight_decay)
criterion = nn.SmoothL1Loss()
scheduler = StepLR(optimizer, step_size, gamma)
'''Model Training'''
for epoch in range(num_epochs):
    my_net.train()
    for batch_features, batch_targets in train_loader:
        outputs = my_net(batch_features)
        loss = criterion(outputs, batch_targets)
        optimizer.zero_grad()  # Clear past gradients
        loss.backward()  # Back propagation, calculate the current gradient
        optimizer.step()  # Update network parameters based on gradients
    # Call scheduler.step() after each epoch to update the learning rate
    scheduler.step()
    # Print the loss of each epoch to monitor the training progress
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

'''Evaluating Model Performance'''
evaluate_model(my_net, train_loader, criterion)
evaluate_model(my_net, test_loader, criterion)
plot_predictions_vs_actuals(my_net, test_loader)
plot_predictions_vs_actuals(my_net, train_loader)

'''Save the model'''
torch.save(my_net.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')

