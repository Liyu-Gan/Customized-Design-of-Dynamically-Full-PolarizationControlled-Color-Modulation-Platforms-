import torch
import math
import csv
import time
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import spectrum as sp
from torch.nn import functional as F
from scipy.interpolate import interp1d
from torch.nn import functional as F
import torch.utils.data
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.animation as animation
from random import randint
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from IPython.display import HTML


# Interpolation function, used to uniformly extract a specified number of data points from the original data
def interpolate_data(data, target_length):
    x_old = np.linspace(0, 1, len(data))
    f = interp1d(x_old, data, kind='linear')
    x_new = np.linspace(0, 1, target_length)
    return f(x_new)

# Color code: red, green, blue, 123
def colorMatrix(color_data,target_length):
    color_matrix=[] #Store 121 parameters corresponding to the color
    j=1
    for cl in Color_Values:
        color_spectrum = color_data[cl]['y']  #Store 121 pigment characteristic absorption spectrum data
        color_features = interpolate_data(color_spectrum, target_length)
        color_matrix.append(color_features)
    return torch.tensor(color_matrix,dtype=torch.float32)



'''Get real data, the first five dimensions are: color, stretching degree, grayscale, a1, a2, a3. The last three dimensions are RGB data.'''
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

def minmaxscaler2(data,max,min):  #MinMaxScaler normalization method
    a=(data - min)/(max-min)
    return a

def inverse_minmaxscaler(data,max,min):#Denormalization function
    d=data*(max-min)+min
    return d

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

    #Combined feature vector
    all_data = np.hstack([norm_color_features, norm_stretch, norm_gray, norm_a1, norm_a2, norm_a3, norm_targets])

    return all_data, keys

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.fc1 = nn.Linear(G_IN_din + 3, 50)  # Input target RGB and random noise
        self.fc2 = nn.Linear(50, 128)
        self.fc3 = nn.Linear(128, 50)
        self.fc4 = nn.Linear(50, 9)  # Output 106-dimensional features

    def forward(self, z, target_rgb):
        x = torch.cat([z, target_rgb], dim=1)  # Merge noise and target RGB
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))  # The output value is between [-1, 1]
        return x
# Model definition
class Bias_Batch_Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, out_dim):
        super(Bias_Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.Dropout(p=0.3))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.BatchNorm1d(n_hidden_3), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4), nn.BatchNorm1d(n_hidden_4), nn.Dropout(p=0.3))
        self.output_layer = nn.Linear(n_hidden_4, out_dim)
        self.bias = nn.Parameter(torch.zeros(out_dim))  # Add additional bias parameters

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output_layer(x) + self.bias  # Add additional bias to the output
        return x
    

def my_loss(input_data):
    # In the dim dimension, add the tesnors inside
    err = torch.sum(F.relu(-input_data)+F.relu(input_data-1))+torch.sum(torch.abs(input_data[:,0]-2/5.0))
    +torch.sum(torch.abs(input_data[:,1]-66/66.0))+torch.sum(torch.abs(input_data[:,2]-2/4.0))
    return err

#The color data is changed from 1 to 121
def color_transform2(color):
    color_labels=torch.from_numpy(np.arange(1.0,NColors+1.0,dtype=np.float32)).to(device)
    color_distance = F.relu(1.0-torch.abs(color_labels-color))
    color_onehot = F.softmax(color_distance*20, 0)
    return torch.matmul(color_onehot.reshape(1,-1),color_matrix)
    
#The total data changes from 6 to 106, and normalizes at the same time
def data_transform(data):
    clist=torch.squeeze(color_transform2(data[0]))
    data1=torch.flatten(torch.cat((clist,data[1:]),0))
    return data1

def norm(data):    
    #Globally normalize the colors
    color_x=data[:-5]#
    color_x=minmaxscaler2(color_x, color_max, color_min)
    color_x=torch.unsqueeze(color_x,0)
    #Normalize the other three parameters by column
    str_x=minmaxscaler2(data[-5],str_max, str_min)
    str_x=torch.unsqueeze(torch.unsqueeze(str_x,0),0)
    gra_x=minmaxscaler2(data[-4],gra_max,gra_min)
    gra_x=torch.unsqueeze(torch.unsqueeze(gra_x,0),0)
    a1_x=minmaxscaler2(data[-3],a1_max,a1_min)
    a1_x=torch.unsqueeze(torch.unsqueeze(a1_x,0),0)
    a2_x=minmaxscaler2(data[-2],a2_max,a2_min)
    a2_x=torch.unsqueeze(torch.unsqueeze(a2_x,0),0)
    a3_x=minmaxscaler2(data[-1],a3_max,a3_min)
    a3_x=torch.unsqueeze(torch.unsqueeze(a3_x,0),0)
    x_un=torch.cat([color_x,str_x,gra_x,a1_x,a2_x,a3_x],dim=1)
    return x_un

#The judgment part determines whether the data generated by the generator meets the range. 
# If it does, it is thrown into the fully connected prediction network to predict the G value;
#  if it does not meet the range, a false G value is given.
def distinguish(input_data):
    data = torch.empty_like(input_data)
    real_input = torch.empty(batch_size,106)
    for j in range(0,batch_size):
        data[j][0]=inverse_minmaxscaler(input_data[j][0],NColors,1)
        data[j][1]=inverse_minmaxscaler(input_data[j][1],str_max,str_min)
        data[j][2]=inverse_minmaxscaler(input_data[j][2],gra_max,gra_min)
        data[j][3]=inverse_minmaxscaler(input_data[j][3],a1_max,a1_min)
        data[j][4]=inverse_minmaxscaler(input_data[j][4],a2_max,a2_min)
        data[j][5]=inverse_minmaxscaler(input_data[j][5],a3_max,a3_min)
        complete_data=data_transform(data[j])
        complete_data_norm=norm(complete_data)
        real_input[j] = torch.squeeze(complete_data_norm)
    return Forward_net(real_input)


SpecRGBPath = r'..\data_value\spc_rgb_dict.json'
RGB_model = r'..\data_value\RGBmodel.pth'
netGPath = r'..\data_value\netG.pth'


'''Color dictionary, feature range'''
ColorPath=r'..\data_value\color.msgz'
'''Reading color data'''
color_data = sp.read_file(ColorPath)

color_max, color_min = 3.184, 0.007
str_max, str_min = 133.0, 67.0
gra_max, gra_min = 5.0, 1.0
a1_max, a1_min = 45.0, -45.0
a2_max, a2_min = 180.0, 0.0
a3_max, a3_min = 45.0, -45.0
RGB_max, RGB_min = 255, 0
NColors = 6
Color_Values = ['Red', 'Green', 'Blue', 'GB', 'RB', 'RG']
batch_size = 16
ngpu = 1 
G_IN_din = 6
Forward_net =  Bias_Batch_Net(106, 128, 217, 152, 93, 3)
lr = 0.001
num_epochs = 50

#Reading JSON Files
with open(SpecRGBPath, 'r') as f:
    data_dict = json.load(f)

scaler = MinMaxScaler(feature_range=(0, 1))
# Perform data processing, model training and evaluation
realdata_array, keys = process_data(data_dict, color_data, target_length=101)
color_matrix=colorMatrix(color_data,101)
realdata = torch.tensor(realdata_array, dtype=torch.float32)
dataloader = torch.utils.data.DataLoader(dataset=realdata, batch_size=batch_size, shuffle=True,drop_last=True)
RGB_data=realdata[:,-3:]

'''Conditional Generative Adversarial Networks'''
# Decide which device we run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")           

'''Generator'''
netG = Generator(ngpu).to(device)

'''Use the forward prediction RGB model as the discriminator'''
# Loading a saved model
Forward_net.load_state_dict(torch.load(RGB_model))
Forward_net.eval()  # Switch to evaluation mode

'''Training the model'''
img_list = []
G_losses = []
D_losses = []
D_x_list = []
D_z_list = []

#Optimizer and loss function
# Initialize BCELoss function Initialize BCELoss function
criterion = nn.MSELoss(reduction='mean')

# Setup Adam optimizers for both G and D
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.9, 0.999))

ideal_RGB = torch.tensor([0, 255, 160], device=device, dtype=torch.float32) / 255  # Reinitialize ideal_RGB
ideal_RGB = ideal_RGB.repeat(batch_size, 1)

'''Pre-training'''
# For each epoch  
for i, data in enumerate(dataloader):
    optimizerG.zero_grad()
    # target_rgb=data[:,-3:]
    noise = torch.randn(ideal_RGB.size(0), G_IN_din, device=device)
    fake_features_rgb = netG(noise,ideal_RGB)
    err_New = my_loss(fake_features_rgb[:, :-3])
    err_New.backward()
    optimizerG.step()
    print(
        f'Step: [{i+1:0>{len(str(len(dataloader)))}}/{len(dataloader)}]',
        f'Loss-New: {err_New.item():.4f}',
        end='\r'
    )

'''Direct training'''

for epoch in range(num_epochs):
    beg_time = time.time()#Returns the current time stamp
    for i, data in enumerate(dataloader):
        optimizerG.zero_grad()
        noise = torch.randn(ideal_RGB.size(0), G_IN_din, device=device)
        output_G = netG(noise,ideal_RGB)#Fake is the discriminator input
        input_D = output_G[:, :-3]
        generate_RGB_G = output_G[:, -3:]
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output_D = distinguish(input_D)

        err_New = my_loss(input_D)
        err_New.backward(retain_graph=True)

        loss_D = criterion(ideal_RGB, output_D)
        loss_D.backward(retain_graph=True)

        loss_G = criterion(ideal_RGB, generate_RGB_G)
        loss_G.backward(retain_graph=True)

        D_G_z = input_D.mean().item()

        optimizerG.step()

        # Output training stats   
        end_time = time.time()
        run_time = round(end_time-beg_time)
        print(
            f'Epoch: [{epoch+1:0>{len(str(num_epochs))}}/{num_epochs}]',
            f'Step: [{i+1:0>{len(str(len(dataloader)))}}/{len(dataloader)}]',
            f'Loss-New: {err_New.item():.4f}',
            f'Loss-D: {loss_D.item():.4f}',
            f'Loss-G: {loss_G.item():.4f}',
            f'G(z): {D_G_z:.4f}',
            f'Time: {run_time}s',
            end='\r'
        )
        # Save Losses for plotting later  
        D_losses.append(loss_D.item())
        G_losses.append(loss_G.item())
        # D_z_list.append(D_G_z)

# Save the generator
torch.save(netG.state_dict(), netGPath)
