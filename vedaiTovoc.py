import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from lxml import etree
from tools import readClassNames


def convertVedai():
    root_folder = 'C:/Users/dambr/obelix/projets/obeDeep/data/vedai/Vedai512_voc/'
    Annot_path = 'C:/Users/dambr/obelix/projets/obeDeep/data/vedai/Annotations512/'
    Image_path = 'C:/Users/dambr/obelix/projets/obeDeep/data/vedai/Vedai512_voc/JPEGImages/'
    image_set_path = 'C:/Users/dambr/obelix/projets/obeDeep/data/vedai/Vedai512_voc/ImageSets/all.txt'

    classes = readClassNames(root_folder + 'vedai.names')
    print(classes)
    set = open(image_set_path, 'r')
    temp = set.read().splitlines()
    for line in temp:
        img = Image.open(Image_path + line + '.png')
        annot_file = open(Annot_path + line + '.txt')
        gt = []
        label = [1,2,4,5,7,8,9,10,11,23,31,201,301]
        for obj in annot_file:
            annot = obj.split(' ')
            annot1 = [int(a) for a in annot[3:]]
            xmin = min(annot1[3:6])
            ymin = min(annot1[7:10])
            xmax = max(annot1[3:6])
            ymax = max(annot1[7:10])
            try:
                gt.append([str(xmin), str(ymin), str(xmax), str(ymax), classes[label.index(annot1[0])]])
            except:
                print(annot1[0])
                print(line)
            
        printVedaiXml(line, root_folder,img.width,img.height,3,gt) 

def printVedaiXml(patch_name,out_folder,windowWidth,windowHeight,C,gt):
    # <annotation/>
    annotation = etree.Element('annotation')
    folder = etree.SubElement(annotation, 'folder')
    folder.text=out_folder

    filename = etree.SubElement(annotation, 'filename')
    filename.text=patch_name + '.png'

    path = etree.SubElement(annotation, 'path')
    path.text=out_folder+ 'JPEPImages/'+ patch_name + '.png'

    # <annotation><source/>
    source = etree.SubElement(annotation, 'source')

    database = etree.SubElement(source, 'database')
    database.text = "vedai"

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
        objects = etree.SubElement(annotation, 'object')

        name = etree.SubElement(objects, 'name')
        name.text = obj[4]

        # <annotation><object/><bndbox/>
        bndbox = etree.SubElement(objects, 'bndbox')

        xmin = etree.SubElement(bndbox, 'xmin')
        xmin.text = obj[0]
        xmax = etree.SubElement(bndbox, 'xmax')
        xmax.text = obj[2]
        ymin = etree.SubElement(bndbox, 'ymin')
        ymin.text = obj[1]
        ymax = etree.SubElement(bndbox, 'ymax')
        ymax.text = obj[3]

    output_file = open(out_folder+ 'Annotations/'+ patch_name + '.xml', 'w')
    #output_file.write('<?xml version="1.0"?>')
    output_file.write(etree.tostring(annotation, encoding='utf8', method='xml', pretty_print=True).decode())
    output_file.close()







