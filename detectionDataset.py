from __future__ import print_function, division
import os
import torch
import pandas as pd
import PIL
from PIL import Image
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
        image = Image.open(img_path)
        label = self.detDataset[idx]["objects"]
        sample = {'image': image, 'label': label, 'name': name}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def show_sample(self, idx):
        ''' Show a dataset sample with plotted ground truth if present'''
        sample = self.__getitem__(idx)
        image, label, name = sample['image'], sample['label'], sample['name']

        # Create figure and axes
        fig,ax = plt.subplots(1)
    
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

        # If image is a tensor (after ToTensor) transform it back just for showing
        if torch.is_tensor(image):
            trans = transforms.ToPILImage()
            img = trans(image)
        else:
            img = image
            
        title = name + " (" + str(img.width) + "," +str(img.height)+ ")"
        ax.imshow(np.asarray(img))

        classes = self.classes
        # Create a Rectangle patch
        for obj in label:
            print("obj")
            print(obj)
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
        trans = transforms.ToTensor()
        # opencv uses BGR and H x W x C
        # numpy image: H x W x C
        # torch image: C X H X W
        return {'image': trans(image),
                'label': label,
                'name' : name}

class ToYolo(object):
    """Convert object in sample to Tensors."""

    def __call__(self, sample):
        image, label, name = sample['image'], sample['label'], sample['name']
        trans = transforms.ToTensor()
        # opencv uses BGR and H x W x C
        # numpy image: H x W x C
        # torch image: C X H X W
        height = image.height
        width = image.width
        new_labels = torch.zeros(30,5)
        for i, obj in enumerate(label):
            obj = obj[-1:]+obj[:-1]
            new_labels[i][0] = obj[0]
            new_labels[i][1] = min(0.99,float(obj[1]/width))
            new_labels[i][2] = min(0.99,float(obj[2]/height))
            new_labels[i][3] = min(0.99,float(obj[3]/width))
            new_labels[i][4] = min(0.99,float(obj[4]/height))
            test = np.array(new_labels)
            if((new_labels[i][1:]>1).sum()>=1):
                print("ca c'est bizarre!")
                print(new_labels[i])
                print(name)
                print(width)
                print(height)
                print(obj)

        
        return {'image': trans(image),
                'label': new_labels,
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

        w,h = image.size
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = image.resize((new_w, new_h))
            
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
    """ Resizes the sample, keeping the aspect ratio consistent,
        and padding the left out areas with the color (128,128,128)

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
        matched to output_size. If int, smaller of image edges is matched
        to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        """
        Args:
            sample (dict{PIL Image, list, string}): sample to be resized.

        Returns:
            dict{PIL Image, list, string}: Resized image.
        """
        image, label, name = sample['image'], sample['label'], sample['name']
        
        img_w, img_h = image.width, image.height
        if isinstance(self.output_size, int):
            if img_h > img_w:
                out_h, out_w = self.output_size * img_h / img_w, self.output_size
            else:
                out_h, out_w = self.output_size, self.output_size * img_w / img_h
        else:
            out_h, out_w = self.output_size

        out_h, out_w = int(out_h), int(out_w)
        new_w = int(img_w * min(out_w/img_w, out_h/img_h))
        new_h = int(img_h * min(out_w/img_w, out_h/img_h))

        resized_image = image.resize((new_w,new_h), PIL.Image.BICUBIC)
        
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
        im = Image.fromarray(np.uint8(canvas))
            
        return {'image': im, 'label': label, 'name': name}

    def __repr__(self):
        if isinstance(self.output_size, int):
            return self.__class__.__name__ + '(output size = {})'.format(self.output_size)
        else:
            return self.__class__.__name__ + '(output size = ({},{}))'.format(self.output_size[0],
                                                                              self.output_size[1])

class RandomVerticalFlip(object):
    """Vertically flip the given sample randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __init__(self, p=0.5):
        assert 0 <= p <= 1
        self.probability = p

    def __call__(self, sample):
        """
        Args:
            sample (dict{PIL Image, list, string}): sample to be flipped.

        Returns:
            dict{PIL Image, list, string}: Randomly flipped image.
        """
        image, label, name = sample['image'], sample['label'], sample['name']
        if random.random() < self.probability:
            tr = transforms.RandomVerticalFlip(1)
            image = tr(image)
            new_label = []
            for obj in label:
                n_cx = obj[0]
                n_cy = image.height - obj[1]
                n_w = obj[2]
                n_h = obj[3]
                new_label.append([n_cx,n_cy,n_w,n_h,obj[4]])
            label = new_label
        return {'image': image, 'label': label, 'name': name}

    def __repr__(self):
        return self.__class__.__name__ + '(probability = {})'.format(self.probability)

class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of a sample

    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        
    def __call__(self, sample):
        """
        Args:
            sample (dict{PIL Image, list, string}): sample to be flipped.

        Returns:
            dict{PIL Image, list, string}: Randomly flipped image.
        """
        image, label, name = sample['image'], sample['label'], sample['name']
        tr = transforms.ColorJitter(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        image = tr(image)
        return {'image': image, 'label': label, 'name': name}

    def __repr__(self):
        return self.__class__.__name__ + '(brightness = {}, contrast = {}, saturation = {}, hue = {})'.format(self.brightness, self.contrast, self. saturation, self.hue)
     
        
