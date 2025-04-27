# Image Classification

## Description
This project tries to classify images based on the scenery that they portray. The photos are broadly classified into 6 groups.
-  Buildings 
-  Forest
-  Glacier
-  Mountain
-  Sea
-  Street

The choice of model is CNN. The model has multiple convolutional layers with increasing filter sizes (from 32 to 1024).
The data is sourced from Kaggle - Intel Image Classification

## Installation
The data can be downloaded from this [link](https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data).

The training data is in the folder named seg_train/seg_train. Inside this folder, there are 6 seperate folders, named buildings, forest, glacier, mountain, sea, and street. And these subfolders contain images corresponding to their subfolder label.

The test data is found under the folder seg_pred/seg_pred.

## Important
The folder contains large files which was pushed using Large File Storage. So after cloning please run flowing command.
```bash
git lfs pull
```
Running predict.py will generate graphs, which is plotted one after the other, but it will only go on for 10 images unless the range is changed in the for loop.
