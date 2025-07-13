import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class FERDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []
        
        # Set base path
        base_path = os.path.join(root_dir, 'train' if mode in ['train', 'val'] else 'test')
        
        # Load images
        for emotion in self.classes:
            emotion_path = os.path.join(base_path, emotion)
            for img_name in os.listdir(emotion_path):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(emotion_path, img_name)
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[emotion])
        
        # Split train/val
        if mode in ['train', 'val']:
            train_img, val_img, train_lbl, val_lbl = train_test_split(
                self.images, self.labels, test_size=0.2, random_state=42, stratify=self.labels
            )
            self.images, self.labels = (train_img, train_lbl) if mode == 'train' else (val_img, val_lbl)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {self.images[idx]}")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def get_data_loaders(root_dir, batch_size=64):
    """Returns train, validation, and test data loaders"""
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_set = FERDataset(root_dir, 'train', train_transforms)
    val_set = FERDataset(root_dir, 'val', val_transforms)
    test_set = FERDataset(root_dir, 'test', val_transforms)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader