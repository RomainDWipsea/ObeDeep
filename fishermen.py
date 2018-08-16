# compilation des tests pour simplification
from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
import cv2
import glob
from os.path import basename
import tools as to

class FishermanDataset(Dataset):
    """Fishermen classification dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.fisherman = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.fisherman)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.fisherman.iloc[idx, 0])
        image = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
        label = self.fisherman.iloc[idx, 1].astype('int')
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

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
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))

        return {'image': img, 'label': label}
    
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'label': label}

class Normalize(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, mean, std):
        assert isinstance(mean, tuple)
        self.mean = mean
        self.std = std
      

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        tr = transforms.Normalize(self.mean, self.std)
        image = tr(image)

        return {'image': image, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

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
    
def calcNorm(dataset):
    m = []
    s = []
    for i in range(len(dataset)):
        sample = dataset[i]
        img = sample['image']
        #print(i, sample['image'].shape, sample['label'].shape
        average_color = [torch.mean(img[i, :, :]) for i in range(img.shape[0])]
        average_color = list( map(float, average_color) )
        m.append(average_color)
        average_std = [torch.std(img[i, :, :]) for i in range(img.shape[0])]
        average_std = list( map(float, average_std) )
        s.append(average_std)
    mean = np.mean(m,0)
    std = np.std(s,0)
    return mean, std
    
    
def show_fisherman(image, label):
    """Show image with landmarks"""
    if image.shape[0]<5:
        image = image.permute((1, 2, 0))
    plt.imshow(image)
    plt.title("Label : " + str(label))
    plt.pause(0.001)  # pause a bit so that plots are updated

def show_fisherman_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, labels_batch = \
            sample_batched['image'], sample_batched['label']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    fig=plt.figure(figsize=(10, 10))
    
    columns = int(batch_size/2)
    rows = int(batch_size/2)
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        print()
        plt.imshow(images_batch[i-1].permute(1, 2, 0))
        plt.title(str(int(labels_batch[i-1])))
    plt.show()
    
def get_vectors(sample, model, layer, device):
    # 1. Load the image with Pillow library
    img = sample['image']
    # 2. Create a PyTorch Variable with the transformed image
    #t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    t_img = Variable(img)
    
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(1,10)

    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        #print(o.data)
        my_embedding.copy_(o.data)

    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)

    # 6. Run the model on our transformed image
    t_img = t_img.float()
    t_img = t_img.to(device)
    model(t_img)

    # 7. Detach our copy function from the layer
    h.remove()

    # 8. Return the feature vector
    return my_embedding
