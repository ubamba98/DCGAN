from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import torch

class DataGenerator(Dataset):

    def __init__(self, X, y):

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
      img = cv2.imread(self.X[idx])
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = preprocess_input(img)
      label = torch.tensor(self.y[idx])
      return img, label

preprocess_input = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64,64)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])