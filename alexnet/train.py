import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from model import AlexNet

# Data transformation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load training and testing datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"
print(f"Using {device}")

# Model initialization
model_name = "AlexNet"
model = AlexNet()
model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Training and testing loop
train_losses = []  # Store average training loss per epoch
test_losses = []   # Store average testing loss per epoch
epochs = 10

for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_train_loss = 0.0
    loop = tqdm(train_loader, total=len(train_loader), leave=False)
    
    # Training phase
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix(train_loss=loss.item())

    # Store the average training loss for this epoch
    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Testing phase
    model.eval()  # Set model to evaluation mode
    running_test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_test_loss += loss.item()

    # Store the average testing loss for this epoch
    avg_test_loss = running_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Testing Loss: {avg_test_loss}")

print('Finished Training')

# Now plot losses per epoch
epochs_range = np.arange(1, epochs + 1)

# Plotting average training and testing losses per epoch
plt.plot(epochs_range, train_losses, label='Training Loss')
plt.plot(epochs_range, test_losses, label='Testing Loss')
plt.title(f"{model_name} Training and Testing Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f'{model_name}_loss.png')
plt.show()
