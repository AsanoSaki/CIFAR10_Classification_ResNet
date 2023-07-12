import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CifarDataset(Dataset):
    def __init__(self, root_dir, csv_file=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        if csv_file:
            self.df = pd.read_csv(csv_file)
            self.class_to_num = {}
            for i, label in enumerate(self.df['label'].unique()):
                self.class_to_num[label] = i
            self.num_to_class = {v: k for k, v in self.class_to_num.items()}
            self.df['encoded'] = self.df['label'].map(self.class_to_num)

        else:
            self.df = pd.DataFrame({
                'id': list(range(1, len(os.listdir(self.root_dir)) + 1)),
                'label': np.zeros(len(os.listdir(self.root_dir))),
                'encoded': np.zeros(len(os.listdir(self.root_dir)))
            })

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, str(self.df.iloc[index, 0]) + '.png')
        image = np.asarray(Image.open(img_path).convert('RGB'))
        y_label = torch.tensor(int(self.df.iloc[index, 2]))
        if self.transform:
            image = self.transform(image)

        return (image, y_label)