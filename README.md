# Arabic-Handwritten-Characters-Recognition-using-CNN

## Introduction to problem
The handwritten character recognition problem,  like in the well-known MNIST dataset, has been a research focus for years due to its importance in automated text processing. Unfortunately, most of the attention has been on English characters. Recognizing handwritten Arabic characters is tough because of the diversity of writing styles, character shapes, and the nature of the script. This project aims to develop a model to classify handwritten Arabic characters by using advanced deep learning models and the Arabic Handwritten Chars dataset.


## Dataset Description
The Arabic Handwritten Characters dataset, introduced in a paper by El-Sawy et al., contains all Arabic characters handwritten by 60 participants of varying ages. The dataset is divided into two splits: a training set with 13,440 images and a testing set with 3,360 images all have the same size (32,32). Each character's label is linked to the corresponding image name, which represents the character's letter name. The following figures show the size of dataset and sample images from the dataset: 

## The Experiment 
We trained our final model using over 6 different architectures and hyperparameters. In the process of training, we have had issues like overfitting and low validation accuracy. We have tried to overcome these challenges through dropout and regularization techniques. 
