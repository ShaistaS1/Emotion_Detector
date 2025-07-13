import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import get_data_loaders
from model import EmotionCNN, initialize_model
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def train_model(root_dir='data/fer2013', num_epochs=50, batch_size=64, learning_rate=0.001):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Initialize model, loss function, and optimizer
    model = initialize_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 
                         lr=learning_rate, 
                         weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, 
                                mode='min', 
                                factor=0.1, 
                                patience=5)

    # Verify dataset path exists
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Dataset directory not found at: {os.path.abspath(root_dir)}")
    
    # Get data loaders
    train_loader, val_loader, _ = get_data_loaders(root_dir, batch_size)
    
    # Create models directory if it doesn't exist
    os.makedirs('data/models', exist_ok=True)

    # Training variables
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Training phase with progress bar
        for images, labels in tqdm(train_loader, 
                                 desc=f'Epoch {epoch+1}/{num_epochs}',
                                 unit='batch'):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Calculate training loss
        train_loss = running_loss / len(train_loader)
        history['train_loss'].append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Print epoch statistics
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'data/models/best_model.pth')
            print('--> Best model saved!')

    # Save final model after training completes
    torch.save(model.state_dict(), 'data/models/final_model.pth')
    print('\nTraining completed! Final model saved.')

    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('data/models/training_history.png')
    plt.show()

if __name__ == '__main__':
    train_model()