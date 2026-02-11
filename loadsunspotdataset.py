import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SunspotDataset(Dataset):
    def __init__(self, root, split):
        self.img_dir = os.path.join(root, split, "images")
        self.lbl_dir = os.path.join(root, split, "labels")
        self.files = sorted(os.listdir(self.img_dir))
        
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        self.lbl_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        lbl = Image.open(os.path.join(self.lbl_dir, name))
        
        return self.img_transform(img), self.lbl_transform(lbl)


# ---------- Usage ----------

root = "sunspotdataset"

train_loader = DataLoader(SunspotDataset(root, "train"), batch_size=32, shuffle=True)
val_loader   = DataLoader(SunspotDataset(root, "validation"), batch_size=32)
test_loader  = DataLoader(SunspotDataset(root, "test"), batch_size=32)

# quick test
images, labels = next(iter(train_loader))
print(images.shape, labels.shape)