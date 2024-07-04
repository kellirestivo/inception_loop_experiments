import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class MonkeyDataset(Dataset):
    def __init__(self, neuronal_data_files, image_cache_path, crop, subsample, scale, time_bins_sum, transform=None):
        self.neuronal_data_files = neuronal_data_files
        self.image_cache_path = image_cache_path
        self.crop = crop
        self.subsample = subsample
        self.scale = scale
        self.time_bins_sum = time_bins_sum
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        data = []
        for file in self.neuronal_data_files:
            with open(file, 'rb') as f:
                data.append(pickle.load(f))
        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = sample['image_path']
        image = Image.open(image_path)
        if self.crop:
            image = image.crop(self.crop)
        if self.scale:
            image = image.resize((int(image.width * self.scale), int(image.height * self.scale)))
        if self.transform:
            image = self.transform(image)
        return image, sample['response']
    
    def get_dataloader(config):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((config['img_mean'],), (config['img_std'],))
        ])
        dataset = MonkeyDataset(
            neuronal_data_files=config['neuronal_data_files'],
            image_cache_path=config['image_cache_path'],
            crop=config['crop'],
            subsample=config['subsample'],
            scale=config['scale'],
            time_bins_sum=config['time_bins_sum'],
            transform=transform
        )
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        return dataloader