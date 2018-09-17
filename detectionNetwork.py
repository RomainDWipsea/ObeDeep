from __future__ import division
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import parseNet as pn
from yoloUtils import *

class DetectionNetwork(nn.Module):
    def __init__(self, cfgfile):
        super(DetectionNetwork, self).__init__()
        self.blocks = pn.parse_cfg(cfgfile)
        self.net_info, self.module_list = pn.create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}   #We cache the outputs for the route layer

        write = 0     #This is explained a bit later
        for i, module in enumerate(modules):        
            module_type = (module["type"])
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
            elif module_type == "route":
                layers = module["layers"]
                print(layers)
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)

            elif  module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]

            elif module_type == 'yolo':        

                anchors = self.module_list[i][0].anchors
                #Get the input dimensions
                inp_dim = int (self.net_info["height"])

                #Get the number of classes
                num_classes = int (module["classes"])

                #Transform 
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:              #if no collector has been intialised. 
                    detections = x
                    write = 1

                else:       
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x
                
        return detections

    def cuda(self, CUDA):
        if(CUDA):
            #device = torch.device("cuda:0")
            for i, module in enumerate(self.module_list):
                self.module_list[i].cuda()#to(device)
        else:
            print("no cuda GPU was found")

    def cpu(self):
        for i, module in enumerate(self.module_list):
            self.module_list[i].cpu()
