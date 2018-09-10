from __future__ import print_function, division
import os
import torch
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
import tools as to
import pickle as pkl
import random

class objDet(Dataset):
    """General object detection dataset."""

    def __init__(self, root_dir, sample_list, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        dir_path = os.path.dirname(os.path.realpath(__file__))
        label_file = to.find_files(root_dir, "*.names")
        if(len(label_file) != 1):
            print("Error, you should have one and only one label file (*.names) in the root directory")
            print("class is not initialised")
            return
            
        classes = to.readClassNames(root_dir + label_file[0])
        file_list = to.read_sample_list(root_dir + "ImageSets/" + sample_list)
  
        VT = []  # Initialize the global directory
        for filename in file_list:
            kept_info = {}
            img_info = to.parseVOCxml(root_dir + "Annotations/" + filename + ".xml", classes)
            kept_info["name"] = img_info["filename"]
            objects = img_info["obj"]
            bndboxes = []
            for obj in objects:
                xmin = int(obj["xmin"])
                xmax = int(obj["xmax"])
                ymin = int(obj["ymin"])
                ymax = int(obj["ymax"])
                cls = int(obj["class"])
                centered_box = to.centerBndbox(xmin, xmax, ymin, ymax)
                centered_box.append(cls)        # adding the class at the end of the object coordinates.
                bndboxes.append(centered_box)
            kept_info["objects"] = bndboxes
            VT.append(kept_info)
            
        self.detDataset = VT
        self.classes = classes
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.detDataset)

    def __getitem__(self, idx):
        name = self.detDataset[idx]["name"]
        img_path = self.root_dir + "JPEGImages/" + self.detDataset[idx]["name"]
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        label = self.detDataset[idx]["objects"]
        sample = {'image': image, 'label': label, 'name': name}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_classes(self):
        return self.classes

    def show_sample(self, idx):
        img_path = self.root_dir + "JPEGImages/" + self.detDataset[idx]["name"]
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        label = self.detDataset[idx]["objects"]
        name = self.detDataset[idx]["name"]

        # Create figure and axes
        fig,ax = plt.subplots(1)
    
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

        # Display the image
        ax.imshow(image)
        classes = self.classes
        # Create a Rectangle patch
        for obj in label:
            objx = obj[0]-int(obj[2]/2)
            objy = obj[1]-int(obj[3]/2)
            ax.add_patch(patches.Rectangle((objx,objy),obj[2],obj[3],linewidth=1,edgecolor='r',facecolor='none'))
            ax.text(objx+3, objy+(obj[3]-3), classes[obj[4]], color='r')
        plt.title(name)
        plt.show()

class ToTensor(object):
    """Convert object in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image/255.0
        # swap color axis because
        # opencv uses BGR and H x W x C
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': label}
