import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image

# Load your trained model
class CNNModel(nn.Module):
    def __init__(self, num_classes=2):  # Update to match your two classes
        super(CNNModel, self).__init__()
        self.model = torchvision.models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# Define the number of classes (binary classification: abnormal and normal)
num_classes = 2

# Initialize the model and load weights
model = CNNModel(num_classes=num_classes)
model.load_state_dict(torch.load(r"D:/Varad_jammer/final_model_Capstone.pth"))
model.eval()

# Check if CUDA is available and move the model to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the frame to match input size of ResNet
    transforms.ToTensor(),  # Convert frame to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet means and stds
])

# Class labels (abnormal and normal)
class_names = ['Abnormal', 'Normal']

# Open the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Convert the frame to RGB (OpenCV uses BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to a PIL image
    pil_image = Image.fromarray(frame_rgb)

    # Preprocess the frame
    input_tensor = transform(pil_image).unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.to(device)

    # Make a prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]

    # Display the predicted class on the frame
    cv2.putText(frame, f"Predicted: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Show the live frame
    cv2.imshow('Webcam Activity Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
