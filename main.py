import os 
import random 
from tqdm import tqdm 

import numpy as np
from sklearn.model_selection import train_test_split 

import cv2

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A 
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt 
import seaborn as sns


class CNN(nn.Module): 

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x).flatten()


class RectsTriags(Dataset): 
    def __init__(self, images_filepaths, transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "rects":
            label = 1.0
        else:
            label = 0.0
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label

train_transform = A.Compose(
    [
        A.SafeRotate(p=1), 
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        ToTensorV2() 
    ]
)

val_transform = A.Compose( 
    [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        ToTensorV2() 
    ]
)

def get_paths():

    rects_path = './rects/'
    triags_path = './triags/'
    rects = os.listdir(rects_path)
    triags = os.listdir(triags_path)
    return [rects_path + i for i in rects] + [triags_path + i for i in triags]
    
random.seed(42)
torch.manual_seed(42)

paths = get_paths()
paths_train, paths_val = train_test_split(paths, test_size=0.33, random_state=42) 
train_data = RectsTriags(paths_train, train_transform)
val_data = RectsTriags(paths_val, val_transform)


train_dataloader = DataLoader(train_data, batch_size=30, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=30, shuffle=True)

model = CNN()
epoches = 1000
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_loss = []
val_loss = []

for epoch in tqdm(range(epoches)): 
    model.train()
    epoch_loss = []

    for X_train, y_train in train_dataloader:
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train.type(torch.FloatTensor))
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
    train_loss.append(sum(epoch_loss)/len(epoch_loss))

    model.eval()
    epoch_loss = []
    with torch.no_grad():
        for X_val, y_val in val_dataloader:
            outputs = model(X_val)
            loss = criterion(outputs, y_val.type(torch.FloatTensor))
            epoch_loss.append(loss.item())
        val_loss.append(sum(epoch_loss)/len(epoch_loss))
        
print('Finished Training')

model.eval()
with torch.no_grad(): 
    correct = 0
    for X_val, y_val in val_dataloader:
        outputs = model(X_val)
        correct += torch.sum(torch.where(outputs > 0.5, 1, 0) == y_val)
    print('val accuracy', correct / len(val_data))

sns.set_theme()
plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(val_loss,label="val")
plt.plot(train_loss,label="train")
plt.xlabel("epoches")
plt.ylabel("Loss")
plt.legend()
plt.show()

while input('Do you want to classify you picture? [y/n] ') == 'y': 
    os.system("python draw.py")
    path =  './myfig/'
    fig_name = os.listdir(path)
    fig = RectsTriags([path+fig_name[0]], val_transform)[0][0]
    fig = torch.unsqueeze(fig, dim=0)
    model.eval()
    with torch.no_grad():
        outputs = model(fig)
        ans = torch.where(outputs > 0.5, 1, 0)
        print("It's a triangle") if ans[0] == 0 else print("It's a rectangle")


