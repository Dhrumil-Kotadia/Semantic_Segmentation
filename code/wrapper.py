import os

import tqdm
import PIL
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from model import UNet

class KITTIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_dir = os.path.join(root_dir, "image_2")
        self.mask_dir = os.path.join(root_dir, "label_2")
        self.transform = transform
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.mask_filenames = sorted(os.listdir(self.mask_dir))
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        mask = (mask > 0).float()  # Convert to binary mask
        return image, mask

class KITTI_Test_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_dir = os.path.join(root_dir, "image_2")
        self.transform = transform
        self.image_filenames = sorted(os.listdir(self.image_dir))
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image,

def save_checkpoint(model, optimizer, epoch, loss, filepath="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at epoch {epoch+1}")

def load_checkpoint(model, optimizer, filepath="checkpoint.pth"):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded from epoch {checkpoint['epoch']+1}")
    return checkpoint['epoch'], checkpoint['loss']

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    with tqdm.tqdm(dataloader, unit="batch") as tepoch:
        for i, (images, masks) in enumerate(tepoch):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            tepoch.set_postfix(loss=loss.item())
    return loss.item()

def test(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    criterion = nn.BCELoss()
    with torch.no_grad():
        for images in dataloader:
            images = images[0].to(device)
            outputs = model(images)
            data = Image.fromarray((outputs[0].squeeze().mul(255).byte().cpu().numpy()))
            data.save(f"output.png")
    avg_loss = total_loss / len(dataloader)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = KITTIDataset(root_dir="/media/storage/lost+found/projects/Road_Segmentation/data/Kitti/raw/training", transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # for epoch in range(10):
    #     loss = train(model, dataloader, criterion, optimizer, device)
    #     print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    #     save_checkpoint(model, optimizer, epoch, loss)
    
    # Load and test model checkpoint
    load_checkpoint(model, optimizer)
    test_dataset = KITTI_Test_Dataset(root_dir="/media/storage/lost+found/projects/Road_Segmentation/data/Kitti/raw/testing", transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test(model, test_dataloader, device)

if __name__ == "__main__":
    main()
