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
            root_dir (string): Directory with all the images.
            sample_list (string) : filename of the imageset to use (usually train.txt, val.txt or test.txt)
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

#    def get_classes(self):
 #       return self.classes

    def show_sample(self, idx):
        sample = self.__getitem__(idx)
        image, label, name = sample['image'], sample['label'], sample['name']

        # Create figure and axes
        fig,ax = plt.subplots(1)
    
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

        # Display the image

        # if image is tensor, then convert it
        if image.shape[2]>4:
            title = name + " (" + str(image.shape[2])+","+str(image.shape[1])+")"
            image = np.transpose(image, (1,2,0))
        else:
            title = name + " (" + str(image.shape[1])+","+str(image.shape[0])+")"
        ax.imshow(image)
        classes = self.classes
        # Create a Rectangle patch
        for obj in label:
            objx = obj[0]-int(obj[2]/2)
            objy = obj[1]-int(obj[3]/2)
            ax.add_patch(patches.Rectangle((objx,objy),obj[2],obj[3],linewidth=1,edgecolor='r',facecolor='none'))
            ax.text(objx+3, objy+(obj[3]-3), classes[obj[4]], color='r')
        plt.title(title)
        plt.show()

class ToTensor(object):
    """Convert object in sample to Tensors."""

    def __call__(self, sample):
        image, label, name = sample['image'], sample['label'], sample['name']
        image = image/255.0
        # swap color axis because
        # opencv uses BGR and H x W x C
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': label,
                'name' : name}

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label, name = sample['image'], sample['label'], sample['name']

        h, w, c = image.shape
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
            
        ratio_w = new_w/w
        ratio_h = new_h/h
        new_label = []
        for obj in label:
            n_cx = int(obj[0]*ratio_w)
            n_cy = int(obj[1]*ratio_h)
            n_w = int(obj[2]*ratio_w)
            n_h = int(obj[3]*ratio_h)
            new_label.append([n_cx,n_cy,n_w,n_h,obj[4]])
        label = new_label

        return {'image': img, 'label': label, 'name': name}

class letterbox_image(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        '''resize image with unchanged aspect ratio using padding'''
        image, label, name = sample['image'], sample['label'], sample['name']
        
        img_w, img_h = image.shape[1], image.shape[0]
        if isinstance(self.output_size, int):
            if img_h > img_w:
                out_h, out_w = self.output_size * img_h / img_w, self.output_size
            else:
                out_h, out_w = self.output_size, self.output_size * img_w / img_h
        else:
            out_h, out_w = self.output_size

        out_h, out_w = int(out_h), int(out_w)
        print(out_h)
        print(out_w)

        new_w = int(img_w * min(out_w/img_w, out_h/img_h))
        new_h = int(img_h * min(out_w/img_w, out_h/img_h))
        print(new_h)
        print(new_w)

        resized_image = cv2.resize(image, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
        
        canvas = np.full((out_h, out_w, 3), 128)

        canvas[(out_h-new_h)//2:(out_h-new_h)//2 + new_h,(out_w-new_w)//2:(out_w-new_w)//2 + new_w,  :] = resized_image

        ratio_w = new_w/img_w
        ratio_h = new_h/img_h
        new_label = []
        for obj in label:
            n_cx = int(obj[0]*ratio_w + (out_w-new_w)/2)
            n_cy = int(obj[1]*ratio_h+(out_h-new_h)/2)
            n_w = int(obj[2]*ratio_w)
            n_h = int(obj[3]*ratio_h)
            new_label.append([n_cx,n_cy,n_w,n_h,obj[4]])
        label = new_label
            
        return {'image': canvas, 'label': label, 'name': name}
