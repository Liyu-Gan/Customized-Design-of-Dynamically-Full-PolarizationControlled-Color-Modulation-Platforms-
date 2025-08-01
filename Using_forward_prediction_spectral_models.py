import spectrum as sp
import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import spectrum as sp  
from scipy.interpolate import interp1d
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import pandas as pd

# Define the interpolation function
def interpolate_data(data, target_length=101):
    original_length = data.shape[0]
    x = np.linspace(0, 1, original_length)
    f = interp1d(x, data, kind='linear')  # Linear interpolation
    x_new = np.linspace(0, 1, target_length)
    return f(x_new)

# Define the inverse interpolation function
def reverse_interpolate_data(interpolated_data, original_length=400):
    interpolated_length = interpolated_data.shape[0]
    x_interpolated = np.linspace(0, 1, interpolated_length)  # The interpolated x-axis
    f_reverse = interp1d(x_interpolated, interpolated_data, kind='linear', bounds_error=False, fill_value="extrapolate")  # Inverse interpolation function
    x_original = np.linspace(0, 1, original_length)  # Original x-axis
    return f_reverse(x_original)  # Returns the deinterpolated data

def process_linear_data(linear_data, color_data):
    features = []
    for data in linear_data:
        parts = data.split('_')
        if parts[0] in color_data:
            color_value = np.array(color_data[parts[0]]['y'])
            color_value_compressed = interpolate_data(color_value, target_length=101) 
            feature_array = [float(part) for part in parts[1:]]  # Extract the remaining parts as features
            feature_vector = np.concatenate([color_value_compressed, feature_array])           
            features.append(feature_vector)
        else:
            print(f"Warning: {parts[0]} not found in color_data")
    features = np.array(features, dtype=object)
    # targets = np.array(targets, dtype=object)
    print(f"Features shape: {features.shape}")
    # print(f"Targets shape: {targets.shape}")
    return features


def minmaxscaler(data, mindata, maxdata):
    data_flat = data.flatten() # Flatten into a one-dimensional array
    data_normalized = (data_flat - mindata) / (maxdata - mindata) 
    data_normalized = data_normalized.reshape(data.shape) 
    return data_normalized

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
    

'''File Path'''
ColorPath=r'..\data_value\color.msgz'
file_path = r'..\data_value\0512正向预测参数.xlsx'
model_path = r'..\data_value\zzbmodel.pth'
output_csv = r'..\data_value\20250707圆偏109.csv'

'''Hyperparameters'''
feature_max_min = {'color': {'min': 0.007, 'max': 3.184}, 
 'stretch': {'min': 67.0, 'max': 133.0}, 
 'gray': {'min': 1.0, 'max': 5.0},
 'a1': {'min': -45.0, 'max': 45.0},
 'a2': {'min': 0.0, 'max': 180.0}, 
 'a3': {'min': -45.0, 'max': 45.0},  
 }
target_min=0.5116666555404663
target_max=109.3960

num =15
my_net = Bias_Batch_Net(106, 226, 394, 372, 220, 101)

'''Read color spectrum data'''
color_data = sp.read_file(ColorPath)

'''Target File'''
df = pd.read_excel(file_path)

'''Data preprocessing'''
# Create an empty list to store the transformed data
formatted_data = []
# Loop through each line and format
for index, row in df.iterrows():
    color = row['颜色']
    val1 = row['拉伸']
    val2 = row['灰度']
    val3 = row['a1']
    val4 = row['a2']
    val5 = row['a3']
    # Splice into required format
    formatted_string = f"{color}_{val1}_{val2}_{val3}_{val4}_{val5}"
    # Add to List
    formatted_data.append(formatted_string)

#Printing Results
for item in formatted_data:
    print(item)

# Calling functions and processing data
features = process_linear_data(formatted_data, color_data)  
color_data_normalized = minmaxscaler(features[:,:101], feature_max_min['color']['min'], feature_max_min['color']['max'])
stretch_data_normalized = minmaxscaler(features[:,101], feature_max_min['stretch']['min'], feature_max_min['stretch']['max'])
gray_data_normalized = minmaxscaler(features[:,102], feature_max_min['gray']['min'], feature_max_min['gray']['max'])
a1_data_normalized = minmaxscaler(features[:,103], feature_max_min['a1']['min'], feature_max_min['a1']['max'])
a2_data_normalized = minmaxscaler(features[:,104], feature_max_min['a2']['min'], feature_max_min['a2']['max'])
a3_data_normalized = minmaxscaler(features[:,105], feature_max_min['a2']['min'], feature_max_min['a2']['max'])

# Concatenate all normalized features
new_features_normalized = np.hstack([color_data_normalized, stretch_data_normalized.reshape(num,1), gray_data_normalized.reshape(num,1), a1_data_normalized.reshape(num,1), a2_data_normalized.reshape(num,1), a3_data_normalized.reshape(num,1)])

'''predict'''
# Creating a Model Instance
# Loading a saved model
my_net.load_state_dict(torch.load(model_path))
my_net.eval()  # Switch to evaluation mode

'''Prediction results'''
# Assume that this is the normalized new feature data, with a shape of (18, 103) and a NumPy array type.
  # You can replace it with the actual normalized new data
new_features_array = new_features_normalized.astype(np.float32)
# Convert NumPy arrays to PyTorch tensors
new_features_tensor = torch.from_numpy(new_features_array) 

# Making predictions
with torch.no_grad():  # Disable gradient calculation to speed up inference and save memory
    predictions = my_net(new_features_tensor)
# Print prediction results
print(predictions)
# We interpolate back from 121 points to 400 points
predictions_reverse = np.array([reverse_interpolate_data(pred, original_length=401) for pred in predictions])

# Denormalize the prediction results
predictions_array = predictions_reverse * (target_max - target_min) + target_min
# Save the prediction results to a CSV file, making sure that the name of each sample corresponds to the input array A (save by column after transposition)
output_df = pd.DataFrame(predictions_array.T, columns=formatted_data)  # After transposition, save by column, each column corresponds to a sample
output_df.to_csv(output_csv, index_label='Output_Dimension')
