Overview
=====
![](https://github.com/JimmyRSlab/Complementarity-aware-Feature-Fusion-for-Aircraft-Detection-via-Unpaired-Opt2SAR-Image-Translation/blob/main/Overall.png)
Regarding the issue of aircraft detection in complex scenarios, we present an aircraft detection method based on optical-SAR complementarity-aware feature fusion.

Installation
======

Please refer to https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix and https://github.com/open-mmlab/mmdetection/tree/2.x.

Datasets
=====
## SFEG
For SFEG, the folder SFEG/datasets/opt2sar_Aircraft consists of six subfolders: trainA, testA, trainB, testB, labels_trainA, and labels_testA. The first four subfolders store source domain and target domain images, while the last two subfolders store source domain image object detection labels (in YOLO format and in a txt file with the same name as the image).
## CFFDNet
For CFFDNet, the folder CFFDNet/datasets/yout_dataset_name consists of three subfolders: annotations, train, and val. Annotations store the coco format JSON labels for the training and validation sets, while optical and sar subfolders are placed under both train and val to store the two modalites images.

Acknowledgments
=====
Our code is inspired by https://github.com/NNNNerd/Triple-I-Net-TINet?tab=readme-ov-file and https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.

Citation
=====
```
@ARTICLE{11036138,
  author={Hu, Jianming and Li, Yuelong and Zhi, Xiyang and Shi, Tianjun and Zhang, Wei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Complementarity-Aware Feature Fusion for Aircraft Detection via Unpaired Opt2SAR Image Translation}, 
  year={2025},
  volume={63},
  number={},
  pages={1-19},
  keywords={Feature extraction;Translation;Aircraft;Optical sensors;Optical imaging;Optical scattering;Radar polarimetry;Training;Atmospheric modeling;Remote sensing;Aircraft detection;complementarity-aware feature fusion;image translation;optical and synthetic aperture radar (SAR);remote sensing scene},
  doi={10.1109/TGRS.2025.3578876}}
```
