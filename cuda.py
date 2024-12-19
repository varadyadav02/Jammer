import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm  # For progress display
import os

# Define directories
train_dir = r"D:/Final_Capstone_Dataset_Split/train"
val_dir = r"D:/Final_Capstone_Dataset_Split/val"
test_dir = r"D:/Final_Capstone_Dataset_Split/test"

# Image transformations (resize, normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match the input size for ResNet18
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize based on ImageNet mean and std
])

# Load datasets
train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Define the CNN model using pretrained ResNet18
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Use pretrained weights explicitly
        # Modify the last layer for our number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Get number of classes (same as the number of folders in train_data)
num_classes = len(train_data.classes)
model = CNNModel(num_classes)

# Check if GPU is available and move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # CrossEntropy is commonly used for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Function to train the model
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    best_accuracy = 0.0  # To keep track of the best validation accuracy

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()  # Set the model to training mode
        running_loss = 0.0
        running_corrects = 0

        # Progress bar for training
        train_progress = tqdm(train_loader, desc="Training", leave=False)

        # Iterate over batches of training data
        for inputs, labels in train_progress:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the parameter gradients

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Update progress bar description
            train_progress.set_postfix(loss=loss.item())

        # Calculate training loss and accuracy
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f'Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        # Validation phase
        val_loss, val_acc = validate_model(model, criterion, val_loader)

        # Save the best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Best model saved with accuracy: {val_acc:.4f}')

    print(f'Training complete. Best validation accuracy: {best_accuracy:.4f}')

# Function to validate the model
def validate_model(model, criterion, val_loader):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    running_corrects = 0

    # Progress bar for validation
    val_progress = tqdm(val_loader, desc="Validating", leave=False)

    # Turn off gradients for validation
    with torch.no_grad():
        for inputs, labels in val_progress:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Update progress bar description
            val_progress.set_postfix(loss=loss.item())

    # Calculate validation loss and accuracy
    val_loss = running_loss / len(val_loader.dataset)
    val_acc = running_corrects.double() / len(val_loader.dataset)
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
    return val_loss, val_acc

# Function to test the model on the test dataset
def test_model(model, test_loader):
    model.eval()  # Set model to evaluation mode
    test_corrects = 0

    # Progress bar for testing
    test_progress = tqdm(test_loader, desc="Testing", leave=False)

    with torch.no_grad():
        for inputs, labels in test_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)

            # Update progress bar description
            test_progress.set_postfix(acc=(test_corrects.double() / len(test_loader.dataset)).item())

    # Calculate test accuracy
    test_accuracy = test_corrects.double() / len(test_loader.dataset)
    print(f'Test Accuracy: {test_accuracy:.4f}')

# Train the model 
train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10)

# Test the model
test_model(model, test_loader)

# Save the final model 
torch.save(model.state_dict(), 'final_model_Capstone.pth')
print("Final model saved as 'final_model.pth'")
