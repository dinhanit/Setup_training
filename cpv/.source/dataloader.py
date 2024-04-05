import torch,cv2
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from param import *

class CustomDataset(Dataset):
    def __init__(self, path_csv,absolute_path="datasets/train"):
        self.path_csv = path_csv
        self.data = pd.read_csv(path_csv)
        self.absolute_path = absolute_path
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data['paths'][idx]
        label = self.data['labels'][idx]
        image = cv2.imread(self.absolute_path+"/"+image_path)
        image = cv2.resize(image,(224,224))
        return torch.tensor(image), torch.tensor(label, dtype=torch.long)
    
data_train = CustomDataset('datasets/train.csv')
data_test = CustomDataset('datasets/test.csv')
TRAINLOADER = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
TESTLOADER = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=True)
