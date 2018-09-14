import cv2
from lxml import etree
from io import StringIO, BytesIO
import torch
import shutil
import glob, os

def slidingWindow(image, stepWidth, stepHeight, windowSize):
    for y in range(0, image.shape[0], stepHeight):
        for x in range(0, image.shape[1], stepWidth):
            yield(x, y, image[y:y+windowSize[1], x:x+windowSize[0]])
            
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
            pp += nn
    return pp
    
def printXml(patch_name,out_folder,x,y,windowWidth,windowHeight,C,gt):
    # <annotation/>
    annotation = etree.Element('annotation')
    folder = etree.SubElement(annotation, 'folder')
    folder.text=out_folder

    filename = etree.SubElement(annotation, 'filename')
    filename.text=patch_name + '.jpg'

    path = etree.SubElement(annotation, 'path')
    path.text=out_folder+patch_name + '.jpg'

    # <annotation><source/>
    source = etree.SubElement(annotation, 'source')

    database = etree.SubElement(source, 'database')
    database.text = "unknown"

    # <annotation><size/>
    size = etree.SubElement(annotation, 'size')

    width = etree.SubElement(size, 'width')
    width.text = str(windowWidth)

    height = etree.SubElement(size, 'height')
    height.text = str(windowHeight)

    depth = etree.SubElement(size, 'depth')
    depth.text = str(C)

    segmented = etree.SubElement(annotation, 'segmented')
    segmented.text="0"
    
    for obj in gt:
        # <annotation><object/>
        objects = etree.SubElement(annotation, 'objects')

        name = etree.SubElement(objects, 'name')
        name.text = "gatherer"

        # <annotation><object/><bndbox/>
        bndbox = etree.SubElement(objects, 'bndbox')

        xmin = etree.SubElement(bndbox, 'xmin')
        xmin.text = obj[0]
        xmax = etree.SubElement(bndbox, 'xmax')
        xmax.text = str(int(obj[0]) + int(obj[2]))
        ymin = etree.SubElement(bndbox, 'ymin')
        ymin.text = obj[1]
        ymax = etree.SubElement(bndbox, 'ymax')
        ymax.text = str(int(obj[1]) + int(obj[3]))

    output_file = open(out_folder+patch_name + '.xml', 'w')
    #output_file.write('<?xml version="1.0"?>')
    output_file.write(etree.tostring(annotation, encoding='utf8', method='xml', pretty_print=True).decode())
    output_file.close()
    
def readVt(filePath):
    #returns a table shaped as [xmin, ymin, width, height]
    total_ground_truth = []
    with open(filePath, "r") as f:
        data = f.readlines()
 
        for line in data:
            coord = line.split()
            gt = [coord[0],coord[1],coord[2],coord[3]]
            total_ground_truth.append(gt)
    return total_ground_truth

def read_sample_list(filePath):
    #returns a table shaped as [xmin, ymin, width, height]
    file_list = []
    with open(filePath, "r") as f:
        data = f.readlines()
        for line in data:
            info = line.split()
            img_name = info[0]
            #nb_obj = info[1]
            file_list.append(img_name)
    return file_list

def extractVt(GT,x,y,windowWidth,windowHeight):
    extractedVt = []
    for bbox in GT:
        rules = [int(bbox[0])-int(bbox[2])/2>x,
                 int(bbox[1])-int(bbox[3])/2>y,
                 int(bbox[2])+int(bbox[0])/2<int(windowWidth/2)+x,
                 int(bbox[3])+int(bbox[1])/2<int(windowHeight/2)+y]
        if all(rules):
            extractedVt.append([str(int(bbox[0])-int(int(bbox[2])/2)-x),
                                str(int(bbox[1])-int(int(bbox[3])/2)-y),
                                bbox[2],
                                bbox[3]])
    return extractedVt 

def parseXML(xmlFile):
    """
    Parse the xml
    """
    with open(xmlFile) as fobj:
        xml = fobj.read()

        
    root = etree.fromstring(xml)
    patch_dict = {}
    objects = []
    for appt in root.getchildren():
        patch_dict[appt.tag] = appt.text
        if appt.tag == "objects":
            obj_dict = {}
            for elem in appt.getchildren():
                obj_dict[elem.tag] = elem.text
                if elem.tag == "bndbox":
                    for data in elem.getchildren():
                        obj_dict[data.tag] = data.text
            objects.append(obj_dict)
    patch_dict['obj'] = objects
    return patch_dict

def parseVOCxml(xmlFile, classes):
    """
    Parse the xml
    """
    with open(xmlFile) as fobj:
        xml = fobj.read()

    root = etree.fromstring(xml)
    sample_dict = {}
    objects = []
    for appt in root.getchildren():
        sample_dict[appt.tag] = appt.text
        if appt.tag == "object":
            obj_dict = {}
            cls = 0
            for elem in appt.getchildren():
                #obj_dict[elem.tag] = elem.text
                if elem.tag == "name":
                    cls = classes.index(elem.text)
                    obj_dict["class"] = str(cls)
                if elem.tag == "bndbox":
                    for data in elem.getchildren():
                        obj_dict[data.tag] = data.text
            objects.append(obj_dict)
    sample_dict['obj'] = objects
    return sample_dict

    
def readClassNames(filename):
    """
    Parse the class name file
    """
    classes = []
    file = open(filename, "r") 
    for line in file:
        #rstrip() removes the \n at the end of lines 
        classes.append(line.rstrip())
    return classes

def extractPatches(img, GT, out_folder, posRatio = 0.5):
    H = img.shape[0]
    W = img.shape[1]
    C = img.shape[2]
    bboxs = GT.as_matrix()

    os.mkdir(root_folder + out_folder)
    patch_imgs_path = root_folder + out_folder

    num = 0
    for bbox in bboxs:
        x_min = bbox[1]-int(bbox[3]/2)
        y_min = bbox[0]-int(bbox[2]/2)
        x_max = bbox[1]+int(bbox[3]/2)
        y_max = bbox[0]+int(bbox[2]/2)
        if x_min > 0 and y_min > 0 and x_max < W and y_max < H:
            patch = img[x_min:x_max, y_min:y_max]
            
            patch_name = original_imName +"_pos_" + str(num) + '.jpg'
            cv2.imwrite(patch_imgs_path + patch_name,patch)
            num=num+1

    #2.2 Extract negative samples
    import random
    i = 0
    n_neg = int(num/posRati - num)
    while i < n_neg:
        #print("iter i = " + str(i))
        x = random.randint(1,W-32)
        y = random.randint(1,H-64)
        isPos = False
        for pos in bboxs:
            if abs(x - pos[1]) < 16 and abs(y-pos[0]) < 32:
                isPos = True
        if not isPos:
            patch_name = original_imName +"_neg_" + str(i) + '.jpg'
            patch = img[y:y+64, x:x+32]
            cv2.imwrite(patch_imgs_path + patch_name,patch)
            i=i+1

import numpy as np
def calcNorm(dataset):
    m = []
    s = []
    for i in range(len(dataset)):
        sample = dataset[i]
        img = sample['image']
        img = img/255
        #print(i, sample['image'].shape, sample['label'].shape
        average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
        m.append(average_color)
        average_std = [img[:, :, i].mean() for i in range(img.shape[-1])]
        s.append(average_std)
    mean = np.mean(m,0)
    std = np.std(s,0)
    return mean, std
            
def confirm():
    """
    Ask user to enter Y or N (case-insensitive).
    :return: True if the answer is Y.
    :rtype: bool
    """
    answer = ""
    while answer not in ["y", "n"]:
        answer = input("OK to push to continue [Y/N]? ").lower()
    return answer == "y"

def trainMyNet(net,trainloader,optimizer,criterion,e,show,device):
    net.train()
    for epoch in range(e):  # loop over the dataset multiple times
        running_loss = 0.0
        #print(len(trainloader))
        for i, data in enumerate(trainloader):
            # get the inputs
            inputs, labels = data['image'], data['label']
        
            #inputs = inputs.transpose(1,3)
            inputs = inputs.float()
            labels = labels.float()
            inputs, labels = inputs.to(device), labels.to(device)
            # print(len(inputs), len(labels), len(data))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            outputs = outputs.view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % show == show-1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / show))
                running_loss = 0.0

    print('Finished Training')
    
def testMyNet(net,testloader, device):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data['image'], data['label']
            #inputs = inputs.transpose(1,3)
            inputs = inputs.float()
            labels = labels.long()
            inputs, labels = inputs.to(device), labels.to(device)
            #net.eval()
            outputs = net(inputs)
            #print(outputs)
            #print(outputs.data)
            predicted = torch.sign(outputs.data)
            #print("pred")
            #print(predicted.view(-1))
            #print(labels.view(-1))
            predicted = predicted.long()
            total += labels.size(0)
            #print(predicted.view(-1) == labels.view(-1))
            correct += (predicted.view(-1) == labels.view(-1)).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy of the network on the %d test images: %d %%' % (total,
        100 * correct / total))
    return accuracy

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        
def find_files(root_dir, pattern):
    os.chdir(root_dir)
    files=[]
    for file in glob.glob(pattern):
        files.append(file)
    return files

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    img = cv2.resize(img, (inp_dim, inp_dim))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def centerBndbox(xmin, xmax, ymin, ymax):
    bnd_width = xmax-xmin
    bnd_height = ymax-ymin
    x_center = xmin+int(bnd_width/2)
    y_center = ymin + int(bnd_height/2)
    return [x_center, y_center, bnd_width, bnd_height]
