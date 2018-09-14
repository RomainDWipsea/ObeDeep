# ObeDeep

This project aims at gathering in one place the most recent deeplearning object detection networks.
All codes should be in pure pytorch code and ready to use in order to be able to compare the outputs with ease.
## Development
- [x] Write the first README.md
- [x] Load an object detection dataset (VOC style)
- [ ] Implement automatic data augmentation
- [ ] Implement Yolov3
- [ ] Implement FasterRCNN
- [ ] Implement Retinanet

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

* Python 3.5.4
* pytorch 0.4.1
* CUDA 9.0
* CUDNN 7

### Installing
* Clone this repository
```
# obeDeep=/path/to/clone/detectron
git clone https://github.com/RomainDWipsea/ObeDeep.git $obeDeep
```

Install Python dependencies:
```
pip3 install -r $obeDeep/requirements.txt
```

* If you want to try out the first classification network, got to $obeDeep and enter : 
```
jupyter notebook extractFishermanFeatures.ipynb
```

## Data
### For Object detection :
All data are stored in Pascal VOC fashion with multiple folders :
- Annotations
  |- img1.xml
  |- 1 annotation file per image in the xml format.
- ImageSets
  |- train.txt
  |- val.txt
  |- test.txt
  |- ...
- JPEGImages
  |- img1.jpg
  |- All images regardless of the training/test set

## Contributing
Contribution are not open at the moment, please contact me if you want to add something : romain.dambreville@irisa.fr

## Versioning
- [ ] No release are available yet.
## Authors

* **Romain Dambreville** 

See also the list of [contributors](https://github.com/RomainDWipsea/ObeDeep/graphs/contributors) who participated in this project.

## License

This project has currently no license, we whould update it soon

## Acknowledgments

* [README.md]{https://gist.github.com/PurpleBooth/109311bb0361f32d87a2#file-readme-template-md} template from 
* [yolo from scratch]{https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/}
* etc

