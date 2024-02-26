import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import numpy as np
from collections import Counter, defaultdict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define your model architecture
model = resnet18(pretrained=True)
num_ftrs = model.fc.in_features
class_names = np.load('class_names.npy')
num_classes = len(class_names)
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# Load the trained weights
model.load_state_dict(torch.load('model.pth'))

# Switch the model to evaluation mode
model.eval()
# Collect all image paths and labels for testing
test_root_dirs = ['./']  # Replace with the actual path to your test images
test_image_paths = []
test_labels = []
for root_dir in test_root_dirs:
    for image_name in os.listdir(root_dir):
        # Extract label from image_name
        if root_dir == 'path_to_your_test_images':  # Replace with the actual path to your test images
            label = 'non-fungi'
        else:
            label = image_name.split('_')[0]  # Adjust this line based on how your labels are included in the image names
        test_image_paths.append(os.path.join(root_dir, image_name))
        test_labels.append(label)
# Create a dictionary where the keys are the class labels and the values are lists of image paths
class_to_image_paths = defaultdict(list)
for path, label in zip(image_paths, labels):
    class_to_image_paths[label].append(path)

# Evaluate the model on 100 images from each class
model.eval()
class_accuracies = {}
for class_label, image_paths in class_to_image_paths.items():
    # Select the first 100 images for this class
    selected_image_paths = image_paths[:100]
    # Create a DataLoader for these images
    dataset = FungiDataset(selected_image_paths, [class_label]*len(selected_image_paths), transform=transform)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    # Evaluate the model on these images
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # Compute the accuracy for this class
    accuracy = correct / total
    class_accuracies[class_label] = accuracy

# Print the accuracy for each class
for class_label, accuracy in class_accuracies.items():
    print(f'Accuracy of {class_label}: {accuracy*100}%')