import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
import os

# ==========================================
# 1. Model Definition
# ==========================================
class TorchCNN(nn.Module):
    """
    A PyTorch implementation of the custom CNN defined in 'model.py'.
    
    Architecture:
    1. Conv2D: 1 input channel -> 8 output channels, kernel size 3x3.
       - No padding (valid padding), stride 1.
       - Input: (Batch, 1, 28, 28) -> Output: (Batch, 8, 26, 26)
    2. ReLU Activation
    3. Flatten: Reshapes (Batch, 8, 26, 26) -> (Batch, 5408)
    4. Linear (FC1): Input 5408 -> Output 128
    5. ReLU Activation
    6. Linear (FC2): Input 128 -> Output 10 (Classes)
    """
    def __init__(self):
        super(TorchCNN, self).__init__()
        
        # Corresponds to Conv2D(in_channels=1, out_channels=8, kernel_size=3)
        self.conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0)
        
        # Calculate flattened dimension: 8 channels * 26 width * 26 height = 5408
        self.flat_dim = 8 * 26 * 26
        
        # Corresponds to LinearLayer(flat_dim, hidden_dim)
        self.fc1 = nn.Linear(self.flat_dim, 128)
        
        # Corresponds to LinearLayer(hidden_dim, num_classes)
        self.fc2 = nn.Linear(128, 10)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # --- Layer 1: Convolution ---
        out = self.conv(x)
        out = self.relu(out)
        
        # --- Flattening ---
        # Reshape the tensor to (Batch_Size, 5408)
        out = out.view(-1, self.flat_dim)
        
        # --- Layer 2: Fully Connected 1 ---
        out = self.fc1(out)
        out = self.relu(out)
        
        # --- Layer 3: Fully Connected 2 (Output) ---
        out = self.fc2(out)
        
        # Note: We do not apply Softmax here because nn.CrossEntropyLoss 
        # includes LogSoftmax internally.
        return out

# ==========================================
# 2. Data Preparation
# ==========================================
def get_data_loaders(batch_size=64):
    """
    Downloads and prepares the MNIST dataset.
    """
    # Define normalization to match standard MNIST statistics
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download training and test data
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader, test_loader

# ==========================================
# 3. Training & Testing Loop
# ==========================================
def train_model():
    # Detect device (Use GPU if available for speed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Initialize model
    model = TorchCNN().to(device)
    
    # Optimizer: SGD with learning rate 0.01 (Matches 'optim.py')
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Loss Function: CrossEntropyLoss (Matches 'SoftMaxOp' + log likelihood)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = get_data_loaders()
    
    # Dictionary to store history for comparison later
    history = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "test_acc": []
    }

    epochs = 40
    
    for epoch in range(1, epochs + 1):
        model.train() # Set model to training mode
        running_loss = 0.0
        
        # --- Training Loop ---
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()           # 1. Clear gradients
            output = model(data)            # 2. Forward pass
            loss = criterion(output, target)# 3. Compute loss
            loss.backward()                 # 4. Backward pass (compute gradients)
            optimizer.step()                # 5. Update weights
            
            running_loss += loss.item()

        # Update learning rate         
        scheduler.step()       
        # Print learning rate
        current_lr = scheduler.get_last_lr()[0]
        print(f"\rEpoch {epoch} [{batch_idx + 1}/{len(train_loader)}] - Loss: {loss.item():.4f} - LR: {current_lr:.6f}", end='')

        # Calculate average training loss for this epoch
        avg_train_loss = running_loss / len(train_loader)
        
        # --- Testing Loop ---
        model.eval() # Set model to evaluation mode
        test_loss = 0
        correct = 0
        with torch.no_grad(): # No need to track gradients for testing
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True) # Get the index of the max probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        avg_test_loss = test_loss / len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)

        print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Accuracy: {accuracy:.2f}%")

        # --- Save Data for Interface ---
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["test_loss"].append(avg_test_loss)
        history["test_acc"].append(accuracy)

    return history

# ==========================================
# 4. Main Execution & Interface Output
# ==========================================
if __name__ == "__main__":
    # Run the training
    history_data = train_model()

    # Save the history to a JSON file.
    # This file serves as the INTERFACE. 
    # Your custom implementation script can read this file to compare results.
    output_filename = "pytorch_history.json"
    with open(output_filename, "w") as f:
        json.dump(history_data, f, indent=4)

    print(f"\n[Interface] Training history saved to '{output_filename}'.")
    print("You can load this JSON file in your plotting script to compare PyTorch vs. Custom Implementation.")