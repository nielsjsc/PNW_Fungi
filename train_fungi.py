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
from collections import Counter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define batch size
batch_size = 32

class FungiDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),  # Convert image to RGB
            transforms.ToTensor()
        ])

        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Convert label to tensor
        if self.transform:
            image = self.transform(image)
        return image, label

# Set up data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets and dataloaders
root_dirs = ['fungi_photos', 'plant_photos']

# Collect all image paths and labels
image_paths = []
labels = []
species_counts = Counter()
for root_dir in root_dirs:
    for image_name in os.listdir(root_dir):
        # Extract label from image_name
        if root_dir == 'PNW_mushrooms/plant_photos':
            label = 'non-fungi'
        else:
            label = image_name.split('_')[0]  # Adjust this line based on how your labels are included in the image names
        # Only add the image if we have less than 100 of this species
        if species_counts[label] < 100:
            image_paths.append(os.path.join(root_dir, image_name))
            labels.append(label)
            species_counts[label] += 1

# Split root_dirs into training, validation, and test sets
# Split image_paths and labels into training, validation, and test sets
train_image_paths, temp_image_paths, train_labels, temp_labels = train_test_split(image_paths, labels, test_size=0.4, random_state=42)  # 60% for training
val_image_paths, test_image_paths, val_labels, test_labels = train_test_split(temp_image_paths, temp_labels, test_size=0.5, random_state=42)  # 20% for validation, 20% for testing

# Count the occurrences of each class in the labels
class_counts = Counter(labels)

# Get the 10 most common classes
most_common_classes = [item[0] for item in class_counts.most_common(10)]

# Filter the training data to only include these classes
filtered_train_image_paths = [path for path, label in zip(train_image_paths, train_labels) if label in most_common_classes]
filtered_train_labels = [label for label in train_labels if label in most_common_classes]

# Filter the validation data to only include the most common classes
filtered_val_image_paths = [path for path, label in zip(val_image_paths, val_labels) if label in most_common_classes]
filtered_val_labels = [label for label in val_labels if label in most_common_classes]

# Create your validation dataset and DataLoader
val_dataset = FungiDataset(filtered_val_image_paths, filtered_val_labels, transform=transform)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

# Create your datasets and DataLoaders
train_dataset = FungiDataset(filtered_train_image_paths, filtered_train_labels, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Get the class names from the training dataset
class_names = train_dataset.label_encoder.classes_

# Save class_names to a file
np.save('class_names.npy', class_names)

# Print the class names
print(class_names)

# Define your model
model = resnet18(pretrained=True)
num_ftrs = model.fc.in_features
num_classes = len(set(train_dataset.labels))  # Number of unique species
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_losses = []
val_losses = []
num_epochs=10
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses.append(train_loss / len(train_loader))

    # Save model after each epoch
    torch.save(model.state_dict(), f'checkpoint_{epoch}.pth')

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_losses.append(val_loss / len(val_loader))

# Plot training and validation loss
torch.save(model.state_dict(), 'model.pth')
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
torch.save(model.state_dict(), 'model.pth')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Compute confusion matrix after training
model.eval()
all_labels = []
all_predictions = []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

cm = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()