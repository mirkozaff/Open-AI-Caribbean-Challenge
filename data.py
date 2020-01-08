from torch.utils import data
import os
import json
import PIL
from PIL import Image
import rasterio
from rasterio.mask import mask
from rasterio.plot import show, reshape_as_image
import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import geopandas as gpd
import random
import time
import cv2

DATA_PATH = '/mnt/data1/zaffaroni_data/competitions_data/DRIVENDATA/'

class Dataset(data.Dataset):
        'Characterizes a dataset for PyTorch'
        def __init__(self, dataset, transform = None, train = True):
            'Initialization'
            self.dataset = dataset         
            self.transform = transform 
            self.train = train
            #self.shuffle_data()

        def shuffle_data(self):   
            random.shuffle(self.dataset)     

        def __len__(self):
            'Denotes the total number of samples'
            return len(self.dataset)

        def __getitem__(self, index):
            'Generates one sample of data'
            # Select sample
            poly = self.dataset[index][2]   
            self.tiff = rasterio.open(self.dataset[index][3])     

            # Load data and get label and id 
            y = self.dataset[index][1]        
            roof_id = self.dataset[index][0]

            # Get roof coords
            roof_coords = torch.tensor(poly.centroid.coords[0], dtype=torch.float)

            # Crop images from GEOtiff
            image_roof, _ = mask(self.tiff, [poly], crop=True, pad=True, filled=True, pad_width=1)
            image_full, _ = mask(self.tiff, [poly], crop=True, pad=True, filled=False, pad_width=1)
            
            image_roof = cv2.cvtColor(image_roof.transpose(1, 2, 0), cv2.COLOR_RGBA2RGB)
            image_full = cv2.cvtColor(image_full.transpose(1, 2, 0), cv2.COLOR_RGBA2RGB)

            image_context = image_full - image_roof
            
            # Convert to PIL Image
            image_roof = Image.fromarray(image_roof)
            image_context = Image.fromarray(image_context)
            #image_full = Image.fromarray(image_full)
            
            # Data agumentation
            if self.train:
                image_roof, image_context = apply_randomTransform(image_roof, image_context)

            if self.transform is not None:
                image_roof = self.transform(image_roof)
                image_context = self.transform(image_context)
                #image_full = self.transform(image_full)
                  
            return image_roof, image_context, roof_coords, y, roof_id 


def load_dataset(data_type, verified = True):

    csv_path = DATA_PATH + 'metadata.csv'
    df = pd.read_csv(csv_path)

    images = df['image'].tolist()
    geojsons = df[data_type].tolist()

    # Classes
    roof_classes = {'concrete_cement' : 0, 'healthy_metal': 1, 'incomplete': 2, 'irregular_metal' : 3, 'other': 4}

    dataset = []
    class_count = [0, 0, 0, 0, 0]

    for gj, img in zip(geojsons, images):

        # Do not consider non human labelled data during test
        if gj is np.nan:
            continue

        gj_path = DATA_PATH + gj
        img_path = DATA_PATH + img
    
        # Load data
        df_roof_geometries = gpd.read_file(gj_path)
        
        if data_type == 'train':
            # Filter unverified
            if verified:
                df_roof_geometries = df_roof_geometries.loc[df_roof_geometries['verified'] == True]
            # Classes to categorical
            df_roof_geometries['roof_material'] = [roof_classes[item] for item in df_roof_geometries['roof_material']]
        else:
            df_roof_geometries['roof_material'] = -1

        for idx, x in zip(df_roof_geometries.groupby(['roof_material']).count().index.values, df_roof_geometries.groupby(['roof_material']).count().id.values):
            class_count[idx] += x

        with rasterio.open(img_path) as tiff:
            tiff_crs = tiff.crs.data
            df_roof_geometries['projected_geometry'] = df_roof_geometries['geometry'].to_crs(tiff_crs)
            df_roof_geometries['img_path'] = img_path
            dataset = dataset + list(df_roof_geometries[['id', 'roof_material', 'projected_geometry', 'img_path']].values)
    
    print(class_count)
    #if data_type == 'train':
    weights = class_weights(class_count)
    print(weights)
    
    return dataset, weights


def apply_randomTransform(image_left, image_right, size=(224,224)):
    
    # Resize
    resize = transforms.Resize(size=size)
    image_left = resize(image_left)
    image_right = resize(image_right)

    # Random crop
    #if random.random() > 0.5:
    #    i, j, h, w = transforms.RandomCrop.get_params(image_left, output_size=(224, 224))
    #   image_left = TF.crop(image_left, i, j, h, w)
    #   image_right = TF.crop(image_right, i, j, h, w)

    # Random horizontal flipping
    if random.random() > 0.5:
        image_left = TF.hflip(image_left)
        image_right = TF.hflip(image_right)

    # Random vertical flipping
    if random.random() > 0.5:
        image_left = TF.vflip(image_left)
        image_right = TF.vflip(image_right)

    # Random rotation
    if random.random() > 0.5:
        angle = random.randint(0, 360)
        image_left = TF.rotate(image_left, angle)
        image_right = TF.rotate(image_right, angle)

    # Random grayscale
    # if random.random() > 0.5:
    #     image_left = TF.to_grayscale(image_left, 3)
    #     image_right = TF.to_grayscale(image_right, 3)

    # Random scale
    if random.random() > 0.5:
        scale = random.uniform(0.2, 2)
        image_left = TF.affine(image_left, angle = 0, translate= (0,0), scale=scale, shear=0)
        image_right = TF.affine(image_right, angle = 0, translate= (0,0), scale=scale, shear=0)

    # Random shear
    if random.random() > 0.5:
        shear = random.randint(-180, 180)
        image_left = TF.affine(image_left, angle = 0, translate= (0,0), scale=1, shear=shear)
        image_right = TF.affine(image_right, angle = 0, translate= (0,0), scale=1, shear=shear)

    return image_left, image_right

def class_weights(class_count):

    base_weights = np.ones(len(class_count))
    class_array = np.array(class_count)
    class_freq = class_array/class_array.sum()
    inverse_class_freq = 1/class_freq

    freq_weights = inverse_class_freq/inverse_class_freq.sum()

    return np.round(base_weights + freq_weights*10, 2).astype(np.float32)

