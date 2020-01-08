import matplotlib
from matplotlib import pyplot as plt
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
import argparse
import math
import numpy as np
from data import Dataset, load_dataset
import csv
from model import RoofEnsemble
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

def predict(model, device, test_loader, writer):
    model.eval()
    pred = []
    with torch.no_grad():
        for (data1, data2, data3, _, id) in test_loader:
            data1, data2, data3 = data1.to(device), data2.to(device), data3.to(device)
            output = model(data1, data2, data3)
            pred = F.softmax(output, dim=1).cpu().numpy()
            writer.writerow((str(id[0]), 
                            str(round(pred[0][0], 1)), 
                            str(round(pred[0][1], 1)), 
                            str(round(pred[0][2], 1)), 
                            str(round(pred[0][3], 1)), 
                            str(round(pred[0][4], 1))))

if __name__ == '__main__':    
    # Use GPUS
    device = torch.device('cuda')
    kwargs = {'num_workers': 0, 'pin_memory': True}

    # Data preprocessing
    preprocessing = transforms.Compose([
                transforms.Resize((224, 224)),  
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    # Dataset loading and splitting
    data, _ = load_dataset(data_type = 'test')
    full_dataset = Dataset(data, preprocessing, train = False)

    # Setting batch data loaders
    test_loader = torch.utils.data.DataLoader(full_dataset, batch_size=1, shuffle=True, **kwargs)

    model = RoofEnsemble().to(device)    
    model.load_state_dict(torch.load('roof_cnn_best.pt'))
    model = nn.DataParallel(model)
    model.eval()
    

    # Save prediction on csv
    with open('prediction.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
        spamwriter.writerow(('id', 'concrete_cement', 'healthy_metal', 'incomplete', 'irregular_metal', 'other'))
        predict(model, device, test_loader, spamwriter)   

    # Make submission compatible with submission format
    prediction_path = 'prediction.csv'
    submission_path = 'submission_format.csv'

    df_pred = pd.read_csv('prediction.csv')
    df_sub = pd.read_csv('submission_format.csv')

    merged = pd.merge(df_sub['id'], df_pred, on = ['id'])
    merged.to_csv('submission.csv', index=False)  
 
    print('File prediction.csv created')
