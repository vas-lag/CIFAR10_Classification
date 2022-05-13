# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 23:53:35 2021

@author: Billy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

batch_size = 128
learning_rate = 1e-3
weight_decay = 5e-4
epochs = 50

transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()])
transform_test = transforms.ToTensor()
    
training_set = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transform_train)
test_set = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(batch_size/2))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        x = self.fc_layer(x)

        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using {} device'.format(device))
    
net = Net().to(device)
    
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

def training_loop(data_loader, net, loss_fn, optimizer):
    dataiter = iter(data_loader)
    size = len(data_loader)
    current_loss = 0.0
    for i in range(size):
        images, labels = dataiter.next()
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        results = net(images)
        loss = loss_fn(results, labels)
        loss.backward()
        optimizer.step()
        
        current_loss += loss.item()
        if i % 2000 == 1999:
            print(f"{i+1} out of {size}   current_loss: {current_loss / (i+1):.4f}")


def testing_loop(data_loader, net, loss_fn):
    dataiter = iter(data_loader)
    size = len(data_loader)
    total_loss = 0.0
    total, correct = 0, 0
    for i in range(size):
        images, labels = dataiter.next()
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            results = net(images)
            total_loss += loss_fn(results, labels)
            _, prediction = torch.max(results, 1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
    print(f"Accuracy: {100 * correct / total}% \n Total loss: {total_loss / size:.4f}")

for epoch_counter in range(epochs):
    print(f"epoch: {epoch_counter+1}")
    training_loop(train_loader, net, loss_fn, optimizer)
    testing_loop(test_loader, net, loss_fn)
        
        
        
        
