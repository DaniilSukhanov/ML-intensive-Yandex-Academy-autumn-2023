import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms

class ImageToImageDataset(Dataset): # Для создания маски
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        
        self.filenames = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.filenames[idx])
        target_path = os.path.join(self.target_dir, self.filenames[idx])

        input_image = Image.open(input_path).convert('L')   # Входное изображение
        target_image = Image.open(target_path).convert('L') # Выходное изображение (маска)

        if self.transform:
            input_image = self.transform(input_image)

        return input_image, target_image

class ImageToNumDataset(Dataset): #Для получения ответа(цифры)
    def __init__(self, answers_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(answers_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = f'img_{self.img_labels.iloc[idx, 0]}.png'
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("L")  # Входное иображение

        label = self.img_labels.iloc[idx, 1] # Ответ

        if self.transform:
            image = self.transform(image)

        return image, label
