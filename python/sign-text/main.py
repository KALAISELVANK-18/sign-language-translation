import numpy as np
import pandas as pd 
import os

for dirname, _, filenames in os.walk(r'D:\khacks'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

train_folder = r'D:\khacks\data'
batch_size =32
transform = transforms.Compose([
  transforms.Grayscale(),
  transforms.Resize((64, 64)),
  transforms.ToTensor()
    ])

dataset = ImageFolder(train_folder, transform=transform)
num_classes = len(dataset.classes)

train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.1)

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset,batch_size=batch_size, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
print(num_classes)

classes = { 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C',
                  13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O',
                  25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y',36:'Z'}
import matplotlib.pyplot as plt
import numpy as np
import torchvision

def imshow(img):
    img = img / 2 + 0.5     
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(train_loader)
images, labels = next(dataiter)
print(labels.unique)

# show images
imshow(torchvision.utils.make_grid(images))

onehot=torch.nn.functional.one_hot(labels, 35)
onehot[0,:]

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten(1,-1)
        self.fc1 = nn.Linear(64*6*6, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 35)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
net = Net()
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
from tqdm import tqdm
epochs=1
train_loss=[]
train_acc=[]
for epoch in range(epochs):  

    running_loss = 0.0
    correct=0
    total=0
    for i, data in tqdm(enumerate(train_loader, 0)):
        inputs, labels = data
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    running_loss /= len(train_loader.dataset)
    epoch_acc = correct / total

    train_loss.append(running_loss)
    train_acc.append(epoch_acc)

    print(f"Epoch: {epoch+1}/{epochs}, Train Loss: {running_loss:.4f}, Train Acc: {epoch_acc:.4f}")
    
print('Finished Training')

m = Net()
m.state_dict()

torch.save(m.state_dict(), 'net.pt')
m_state_dict = torch.load('net.pt')
new_m = Net()
new_m.load_state_dict(m_state_dict)

class_labels = list(classes.values())

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct//total}%')

import matplotlib.pyplot as plt
from PIL import Image

filename = r'D:\khacks\data\K\4.jpg'
img = Image.open(filename)
img = transform(img)
img = img.unsqueeze(0)  

plt.imshow(img.squeeze().numpy(), cmap='gray')
plt.show()

with torch.no_grad():
    outputs = net(img)
    _, predicted = torch.max(outputs, 1)
    predicted_label = class_labels[predicted.item()]

print("Predicted Label: ", predicted_label)