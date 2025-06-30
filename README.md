Installation
======

Please refer to https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix and https://github.com/open-mmlab/mmdetection/tree/2.x.

Datasets
=====
## SFEG
For SFEG, the folder SFEG/datasets/opt2sar_Aircraft consists of six subfolders: trainA, testA, trainB, testB, labels_trainA, and labels_testA. The first four subfolders store source domain and target domain images, while the last two subfolders store source domain image object detection labels (in YOLO format and in a txt file with the same name as the image).
## CFFDNet
For CFFDNet, the folder CFFDNet/datasets/yout_dataset_name consists of three subfolders: annotations, train, and val. Annotations store the coco format JSON labels for the training and validation sets, while optical and sar subfolders are placed under both train and val to store the two modalites images.

