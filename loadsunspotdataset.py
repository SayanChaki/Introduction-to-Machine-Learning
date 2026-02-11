import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

class SafeSunspotDataset(Dataset):
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
        img_path = os.path.join(self.img_dir, name)
        lbl_path = os.path.join(self.lbl_dir, name)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Image corrompue: {img_path} | {e}")
            img = Image.new("RGB", (256, 256))
        try:
            lbl = Image.open(lbl_path).convert("L")
        except Exception as e:
            print(f"Label corrompu: {lbl_path} | {e}")
            lbl = Image.new("L", (256, 256))
        
        return self.img_transform(img), self.lbl_transform(lbl)

if __name__ == "__main__":
    root = "/Users/alizeeguitton/Desktop/SunspotsYoloDataset"

    train_loader = DataLoader(SafeSunspotDataset(root, "train"),
                              batch_size=32, shuffle=True, num_workers=0)

    # tester une seule batch
    images, labels = next(iter(train_loader))
    print(images.shape, labels.shape)
