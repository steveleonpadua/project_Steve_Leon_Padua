
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import torchvision.transforms as tt
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import config


class Image_Folder:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = tt.Compose([tt.ToTensor(),tt.Resize((config.resize_x, config.resize_y))])
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            class_dir = os.path.join(root_dir, cls_name)
            if os.path.isdir(class_dir):
                images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) 
                          if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                self.image_paths.extend(images)
                self.labels.extend([self.class_to_idx[cls_name]] * len(images))
        
        print(f"Found {len(self.image_paths)} images in {root_dir}")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class CustomDataLoader:
    def __init__(self, dataset, batch_size=config.batch_size, shuffle=False, device='cuda', has_labels=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.indices = list(range(len(dataset)))
        self.has_labels = has_labels  
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.dataset):
            raise StopIteration
        
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size

        batch_images = []
        batch_labels = []
        for idx in batch_indices:
            item = self.dataset[idx]
            if self.has_labels:
                image, label = item  
                batch_labels.append(label)
            else:
                image = item 
            batch_images.append(image)

        batch_images = torch.stack(batch_images).to(self.device)
        if self.has_labels:
            batch_labels = torch.tensor(batch_labels).to(self.device)
            return batch_images, batch_labels
        else:
            return batch_images

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size






