from __future__ import division
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import parseNet as pn
from yoloUtils import *

class DetectionNetwork(nn.Module):
    """General object detection network."""
    def __init__(self, cfgfile, cuda):
        """
        Args:
            cfgfile (string): Absolute path of the cfg file
            cuda (bool) : true if torch.cuda.is_available() is true
        """
        super(DetectionNetwork, self).__init__()
        self.blocks = pn.parse_cfg(cfgfile)
        self.net_info, self.module_list = pn.create_modules(self.blocks)
        self.cuda = cuda
        self.toDevice(cuda)
        self.seen = 0

    def forward(self, x, target=None):
        """
        Args:
            x (torch.Tensor): intput data
            target (torch.Tensor) : ground truth (for the training phase)

        Returns:
            torch.Tensor : detections int the format [Batch x 10647(nb_bbox) x 85(nb_class + 5)]
        """
        modules = self.blocks[1:]
        outputs = {}   #We cache the outputs for the route layer

        write = 0     
        for i, module in enumerate(modules):        
            module_type = (module["type"])
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
            elif module_type == "route":
                layers = module["layers"]
                #print(layers)
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
                anchor_step = 2
                #print("num_anchors = %f" % len(anchors))
                #Get the input dimensions
                inp_dim = int (self.net_info["height"])
                noobject_scale = 1
                object_scale = 5
                thresh = 0.6
                seen = 0
                coord_scale = 1
                class_scale = 1
                
                #Get the number of classes
                num_classes = int (module["classes"])
                print("numclasses = %d" % num_classes)
                #x = x.data
                if self.training:
                    nB = x.size(0)
                    nA = int(len(anchors))#self.module_list[i][0].num_anchors #self.num_anchors
                    nC = num_classes
                    nH = x.size(2)
                    nW = x.size(3)

                    # read 2.1 in YOLOv3 paper
                    x = x.view(nB, nA, (5+nC), nH, nW)
                    xp = F.sigmoid(x.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
                    y = F.sigmoid(x.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
                    w  = x.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
                    h  = x.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
                    conf = F.sigmoid(x.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))
                    cls  = x.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long().cuda()))
                    cls  = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)
    
                    pred_boxes = torch.cuda.FloatTensor(4, nB*nA*nH*nW)
                    grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
                    grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
                    anchor_w = torch.Tensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([0])).cuda()
                    anchor_h = torch.Tensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([1])).cuda()
                    anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
                    anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)

                    pred_boxes[0] = xp.data.view(nB*nA*nH*nW) + grid_x
                    pred_boxes[1] = y.data.view(nB*nA*nH*nW) + grid_y
                    pred_boxes[2] = torch.exp(w.data.view(nB*nA*nH*nW)) * anchor_w
                    pred_boxes[3] = torch.exp(h.data.view(nB*nA*nH*nW)) * anchor_h
                    pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,4))

                    nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf,tcls = build_targets(pred_boxes, target.data, anchors, nA, nC, \
                                                                   nH, nW, noobject_scale, object_scale, thresh, seen)
                    cls_mask = (cls_mask == 1)
                    nProposals = int((conf > 0.25).sum().data[0])

                    tx    = Variable(tx.cuda())
                    ty    = Variable(ty.cuda())
                    tw    = Variable(tw.cuda())
                    th    = Variable(th.cuda())
                    tconf = Variable(tconf.cuda())
                    tcls  = Variable(tcls.view(-1)[cls_mask.view(-1)].long().cuda())
                    
                    coord_mask = Variable(coord_mask.cuda())
                    conf_mask  = Variable(conf_mask.cuda().sqrt())
                    cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,nC).cuda())
                    cls = cls[cls_mask].view(-1, nC)

                    loss_x = coord_scale * nn.MSELoss(size_average=False)(xp*coord_mask, tx*coord_mask)/2.0
                    criterion = nn.MSELoss(size_average=False)
                    loss_y = coord_scale * criterion(y*coord_mask, ty*coord_mask)/2.0
                    loss_w = coord_scale * nn.MSELoss(size_average=False)(w*coord_mask, tw*coord_mask)/2.0
                    loss_h = coord_scale * nn.MSELoss(size_average=False)(h*coord_mask, th*coord_mask)/2.0
                    loss_conf = nn.MSELoss(size_average=False)(conf*conf_mask, tconf*conf_mask)/2.0
                    loss_cls = class_scale * nn.CrossEntropyLoss(size_average=False)(cls, tcls)
                    loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
                else:
                    x = predict_transform(x, inp_dim, anchors, num_classes, self.cuda)
                    if not write:              #if no collector has been intialised. 
                        detections = x
                        write = 1
                    else:       
                        detections = torch.cat((detections, x), 1)
            else:
                print('unknown type %s' % (block['type']))
                
            outputs[i] = x
        if self.training:
            return loss
        else:
            return detections

    def toDevice(self, CUDA):
        """ Send the model to either GPU or CPU according to CUDA value.

        Args:
            CUDA (bool): true if gpu is available

        Returns:
            none
        """
        if(CUDA):
            #device = torch.device("cuda:0")
            self.cuda = CUDA
            for i, module in enumerate(self.module_list):
                self.module_list[i].cuda()
        else:
            self.cuda = False
            for i, module in enumerate(self.module_list):
                self.module_list[i].cpu()
            print("no cuda GPU was found")


    def load_weights(self, weightfile):
        """ load a weights file in the existing network (compatible with yolov3)

        Args:
            weightfile (string): Absolute path to the weight file

        Returns:
            none
        """
        #Open the weights file
        fp = open(weightfile, "rb")

        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype = np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            
            #If module_type is convolutional load weights
            #Otherwise ignore.
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                    
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

    def output(self, img, threshold = 0.5, nms = 0.3):
        """ Infer detections on an image

        Args:
            img (PIL Image) : input image
            threshold (float) : confidence threshold
            nms (float) : non maximum suppression threshold

        Returns:
            None
        """
        assert 0 <= threshold <= 1
        assert 0 <= nms <= 1
        
        self.module_list.eval()
        img_ = img[np.newaxis,:,:,:]
        if(self.cuda):
            img_ = img_.cuda()

        prediction = self.forward(Variable(img_, volatile = True))
        prediction = write_results(prediction, threshold, 80, nms)
        print(prediction.shape)
        print(prediction)
        print(img.shape)

        prediction[0, [1,3]] = torch.clamp(prediction[0, [1,3]], 0.0, img.shape[1])
        prediction[0, [2,4]] = torch.clamp(prediction[0, [2,4]], 0.0, img.shape[2])
        print(prediction[0:])

        # Create figure and axes
        fig,ax = plt.subplots(1)
    
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

        # If image is a tensor (after ToTensor) transform it back just for showing
        if torch.is_tensor(img):
            trans = transforms.ToPILImage()
            img = trans(img)
        title = "test" + " (" + str(img.width) + "," +str(img.height)+ ")"
        ax.imshow(np.asarray(img))

        # Create a Rectangle patch
        bndBoxes = prediction[:,[1,2,3,4,7]]
        bndboxes = bndBoxes.int()
        print(bndBoxes)
        for box in bndBoxes:
            width = int(box[2]) - int(box[0])
            height = int(box[3]) - int(box[1])
            ax.add_patch(patches.Rectangle((int(box[0]),int(box[1])),width,height,linewidth=1,edgecolor='r',facecolor='none'))
            ax.text(box[0]+3, box[1]+(height-3), str(int(box[4])), color='r')

        plt.title(title)
        plt.show()

    def train():
        if not os.path.exists(backupdir):
            os.mkdir(backupdir)
    
        ###############
        torch.manual_seed(seed)

        if self.cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus
            torch.cuda.manual_seed(seed)

        #model = Darknet(cfgfile)
        region_loss = model.loss
