import torch
from pathlib import Path
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class KariRoadsDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=False):
        self.root = Path(root)
        self.train = train
        task = 'train' if train else 'val'
        self.img_dir = self.root/f'{task}'/'images'
        self.img_files = sorted(self.img_dir.glob('*.png'))
        self.transforms = get_transforms(train)

    def __getitem__(self, idx):
        img_file= self.img_files[idx].as_posix()
        label_file = img_file.replace('images', 'png_labels')
        img = cv2.imread(img_file).astype(np.float32) / 255.0   # 0-1 normalization
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)              # RGB order for compatibility with others
        label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)    # class ids: 0,1,2,...,9 
        img, label = self.transforms(img, label)
        return img, label, img_file

    def __len__(self):
        return len(self.img_files)
    
    def get_class_names(self):
        return ['background', 'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'path', 'under construction', 'train guideway', 'airplay runway']
    
class ImageAug:
    def __init__(self, train):
        if train:
            self.aug = A.Compose([A.RandomCrop(512, 512),
                                  A.HorizontalFlip(p=0.5),
                                  A.Affine(scale=(0.8, 1.2), rotate=(-10, 10), translate_percent=(-0.1, 0.1), p=0.3),
                                  A.RandomBrightnessContrast(p=0.3),
                                  ToTensorV2()])
        else:
            self.aug = ToTensorV2()

    def __call__(self, img, label):
        transformed = self.aug(image=img, mask=np.squeeze(label))
        return transformed['image'], transformed['mask']

def get_transforms(train):
    transforms = ImageAug(train)
    return transforms