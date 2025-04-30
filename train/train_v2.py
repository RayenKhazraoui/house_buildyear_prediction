import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch.multiprocessing


torch.multiprocessing.set_start_method("spawn", force=True)

class PreprocessedBuildingDataset(Dataset):
    def __init__(self, dataframe, preprocessed_dir, min_year=1900, max_year=2024, bin_width=20):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing metadata (e.g., 'bouwjaar').
            preprocessed_dir (str): Path to the folder with preprocessed .pt files.
            min_year (int): The minimum year for binning.
            max_year (int): The maximum year for binning.
            bin_width (int): The width of each bin in years.
        """
        self.df = dataframe.reset_index(drop=True)
        self.preprocessed_dir = preprocessed_dir
        self.min_year = min_year
        self.max_year = max_year
        self.bin_width = bin_width
        self.num_bins = ((max_year - min_year) // bin_width) + 1

        # Add a 'label' column to the DataFrame
        self.df['label'] = ((self.df['bouwjaar'] - self.min_year) // self.bin_width).clip(0, self.num_bins - 1)
        
        # Extract labels as a NumPy array
        self.labels = self.df['label'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            image (torch.Tensor): Preprocessed image tensor.
            label (torch.Tensor): Class label.
        """
        preprocessed_path = f"{self.preprocessed_dir}/{idx}.pt"
        try:
            data = torch.load(preprocessed_path, weights_only=True)
            image, label = data['image'], data['label']
        except Exception as e:
            print(f"Error loading file {preprocessed_path}: {e}")
            return torch.zeros((3, 224, 224)), -1  # Return placeholder in case of error

        return image, torch.tensor(label, dtype=torch.long)
    


# Load Data
df_loaded = pd.read_pickle('filtered_data_distance_less_25.pkl')[:-1]
preprocessed_dir = "preprocessed_v2"
full_dataset = PreprocessedBuildingDataset(df_loaded, preprocessed_dir)

# Split Dataset
train_size = int(0.8 * len(full_dataset))
val_size = (len(full_dataset) - train_size) // 2
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size]
)



from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torchvision.models import ResNet18_Weights

import numpy as np

from sklearn.utils.class_weight import compute_class_weight


def create_model(num_classes=full_dataset.num_bins):
    """
    Create a ResNet-18 model with layers 3 and 4 unfrozen, and a customized final layer.

    Args:
        num_classes (int): Number of output classes (e.g., number of bins).

    Returns:
        torch.nn.Module: A ResNet-18 model with customized final layer.
    """
    # Load ResNet-18 with pretrained weights
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layers 3 and 4
    # for param in model.layer3.parameters():
    #     param.requires_grad = True


    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Unfreeze the final fully connected layer
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


# Create the model
model = create_model()

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Compute class weights
class_counts = np.bincount(full_dataset.labels)
class_weights = compute_class_weight(
    class_weight='balanced', classes=np.arange(full_dataset.num_bins), y=full_dataset.labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Loss function with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer: Include unfrozen layers (layers 3, 4, and fc)
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),  # Only update trainable parameters
    lr=1e-3
)



from tqdm import tqdm

def train_model(model, train_loader, val_loader, epochs=15):
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        print(f'Epoch {epoch+1}/{epochs}')
        
        # Progress bar for training
        train_progress = tqdm(train_loader, desc="Training", leave=False)
        for inputs, labels in train_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_progress.set_postfix(loss=loss.item())
        
        train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        
        # Progress bar for validation
        val_progress = tqdm(val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for inputs, labels in val_progress:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                val_progress.set_postfix(loss=loss.item())
        
        # Metrics
        val_loss = val_loss / len(val_loader)
        val_acc = correct / len(val_loader.dataset)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            
    return history


import pickle

if __name__ == "__main__":

    import torch.multiprocessing
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Create DataLoaders
    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # # Define model
    # model = create_model(num_classes=full_dataset.num_bins)  # Use the same function you used before

    # # Load the saved model weights
    # model.load_state_dict(torch.load('best_model.pth'))
    # model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # # Define optimizer and criterion
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Adjust learning rate if needed
    # criterion = nn.CrossEntropyLoss()  # Use the same criterion as before

    print('start training')

    # Start training
    history = train_model(model, train_loader, val_loader, epochs=15)

    # Save history as a .pkl file
    with open("training_history_v2.pkl", "wb") as f:
        pickle.dump(history, f)

    print("Training history saved to 'training_history.pkl'")

    # train_loader = DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    # )

    # for images, labels in train_loader:
    #     print(images.shape, labels.shape)
