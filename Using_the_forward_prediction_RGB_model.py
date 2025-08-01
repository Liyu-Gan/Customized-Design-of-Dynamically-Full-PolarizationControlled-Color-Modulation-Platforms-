import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def plot_predictions_vs_actuals_for_sample(model, train_loader, test_loader, sample_key):
    model.eval()

    # Find samples in the training set and test set
    sample_found_in_train = False
    sample_found_in_test = False
    sample_data_loader = None
    sample_data_loader_name = ""
    sample_idx_in_loader = None

    # Find samples in the training set
    with torch.no_grad():
        for i, (inputs, targets, keys) in enumerate(train_loader):
            if sample_key in keys:
                sample_found_in_train = True
                sample_data_loader = train_loader
                sample_data_loader_name = "Training Set"
                sample_idx_in_loader = keys.index(sample_key)
                break

    # If no sample is found, continue to check the test set
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
        if sample_key in keys:
            sample_idx_in_loader = keys.index(sample_key)
            
            outputs = model(inputs)
            predicted_spectrum = outputs[sample_idx_in_loader].detach().numpy()
            actual_spectrum = targets[sample_idx_in_loader].detach().numpy()

            predicted_RGB = predicted_spectrum * 255
            actual_RGB = actual_spectrum * 255
            break  # Exit the loop after finding the sample

    # Plot the predicted and actual colors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plotting predicted RGB
    ax1.imshow([[predicted_RGB.astype(int)]])
    ax1.set_title(f"Predicted RGB\n{sample_key}")
    ax1.axis('off')
    ax1.text(0.5, 0.5, f"RGB: {predicted_RGB.astype(int)}", color="black", fontsize=12, ha="center", va="center", transform=ax1.transAxes)

    # Draw actual RGB
    ax2.imshow([[actual_RGB.astype(int)]])
    ax2.set_title(f"Actual RGB\n{sample_key}")
    ax2.axis('off')
    ax2.text(0.5, 0.5, f"RGB: {actual_RGB.astype(int)}", color="black", fontsize=12, ha="center", va="center", transform=ax2.transAxes)

    plt.tight_layout()
    plt.show()

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


# Creating a Model Instance
my_net =  Bias_Batch_Net(106, 128, 217, 152, 93, 3)

RGB_model = r'..\data_value\RGBmodel.pt'
# Loading a saved model
my_net.load_state_dict(torch.load(RGB_model))
my_net.eval()  # 切换到评估模式

# Plot the predicted and actual spectra of the specified sample
# Example usage
sample_key = "Red_133_3_-45_0_45"  # Replace with the key name of the sample you want to view
plot_predictions_vs_actuals_for_sample(my_net, train_loader, test_loader, sample_key)
