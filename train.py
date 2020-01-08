import matplotlib
from matplotlib import pyplot as plt
import json
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
import argparse
import math
import numpy as np
from data import Dataset, load_dataset
import os
from model import RoofEnsemble, RoofNet

# Set cuda 10.1
os.environ['PATH'] = '{}:{}'.format('/usr/local/cuda-10.1/bin', os.environ['PATH'])
os.environ['LD_LIBRARY_PATH'] = '{}:{}'.format('/usr/local/cuda-10.1/lib64', os.environ['LD_LIBRARY_PATH'])

print(os.system('nvcc --version'))

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'

def train(args, model, device, train_loader, optimizer, epoch, weights=None):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data1, data2, data3, target, _) in enumerate(train_loader):
        data1, data2, data3, target = data1.to(device), data2.to(device), data3.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data1, data2, data3)
        #output = model(data1)
        #loss = F.nll_loss(output, target)     
        loss = F.cross_entropy(output, target, weight=weights, reduction='mean')
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data1), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
        
    avg_train_loss = train_loss / len(train_loader) # reduction='mean', then divide by batch number
    train_acc = 100. * correct / len(train_loader.dataset)

    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    avg_train_loss, correct, len(train_loader.dataset), train_acc))  

    return avg_train_loss, train_acc

def test(args, model, device, test_loader, weights=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data1, data2, data3, target, _) in enumerate(test_loader):
            data1, data2, data3, target = data1.to(device), data2.to(device), data3.to(device), target.to(device)
            output = model(data1, data2, data3)
            #output = model(data1)
            #test_loss += F.nll_loss(output, target, reduction='sum').item() 
            loss = F.cross_entropy(output, target, weight=weights, reduction='sum')
            test_loss += loss.item() 
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    avg_test_loss = test_loss / len(test_loader.dataset) # reduction='sum', then divide by the full dataset
    test_acc = 100. * correct / len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    avg_test_loss, correct, len(test_loader.dataset), test_acc))   

    return avg_test_loss, test_acc

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch RoofNet test')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')                            
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    #print(torch.cuda.is_available())
    
    #torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    # Data preprocessing
    preprocessing = transforms.Compose([
                transforms.Resize((224, 224)),  
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    # Dataset loading and splitting
    data, weights = load_dataset(data_type = 'train', verified = True)
    validation_split = .05
    random_seed= 42
    shuffle_dataset = True

    # Creating data indices for training and validation splits:
    dataset_size = len(data)
    split = int(np.floor(validation_split * dataset_size))
    # Shuffle data
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(data)
    train_data, val_data = data[split:], data[:split]   
    train_dataset, val_dataset = Dataset(train_data, preprocessing), Dataset(val_data, preprocessing, train = False)
    #full_dataset = Dataset(data, preprocessing)
    #train_size = int(0.8 * len(full_dataset))
    #val_size = len(full_dataset) - train_size
    #train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Setting batch data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, **kwargs)

    # Creating model and loading on gpu
    #model = RoofNet().to(device)
    model = RoofEnsemble().to(device)
    print(model)
    model = nn.DataParallel(model) #As multi-gpu in Keras

    # Optimizer
    #optimizer = optim.ASGD(model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
    #optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # Class weights
    weights = torch.from_numpy(weights).to(device)

    best = 100
    writer = SummaryWriter(log_dir='./logs/5', max_queue=1) # max_queue=1 to flush data at every add

    # Training and testing phase
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(args, model, device, train_loader, optimizer, epoch, weights)
        val_loss, test_acc = test(args, model, device, val_loader, weights)
        if val_loss < best:
            torch.save(model.module.state_dict(), 'roof_cnn_best.pt')
            best = val_loss
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', val_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Acc/test', test_acc, epoch)


    # Saving model
    if (args.save_model):
        torch.save(model.module.state_dict(), 'roof_cnn.pt')
        
if __name__ == '__main__':
    main()