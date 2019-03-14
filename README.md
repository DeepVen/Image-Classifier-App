
#  Pytorch Image Classifier as a command line application
Udacity Data Scientist Nanodegree portfolio project to develop image classifier in Pytorch and make it available as command line application to train model and make predictions. 

## Table of Contents

- [Project Overview](#projectoverview)
- [Technical Overview](#technicaloverview)
- [Execution](#Execution)
- [Requirements](#requirements)



***

<a id='projectoverview'></a>

## 1. Project Overview

In this project, we build and train a deep neural network model on the flower data set and make it available as a command line application which can then be integrated into web app or other application as required. 

<a id='technicaloverview'></a>

## 2. Technical overview:
We develop a command line app using argparse library to allow user to provide various inputs/parameters to train and make inferences on the DL model. Our model as such is a deep neural network built in Pytorch. 

Key python files:
prepare.py does preprocessing of data to get it ready for training.
train_model.py contains model definition, training and validation functions.
load_save_model.py is the utility function to save model to file system and to load checkpoint file as required.
predict.py defines arguments for inference and also contains data preprocessing and prediction definition.

<a id='Execution'></a>
## 3. Execution instructions:
(as provided by Udacity)
Train a new network on a data set with train.py

Basic usage: python train.py data_directory
- Prints out training loss, validation loss, and validation accuracy as the network trains
- Options:
- Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
- Choose architecture: python train.py data_dir --arch "vgg13"
- Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
- Use GPU for training: python train.py data_dir --gpu
- Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint
- Options:
- Return top KK most likely classes: python predict.py input checkpoint --top_k 3
- Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
- Use GPU for inference: python predict.py input checkpoint --gpu


<a id='requirements'></a>

## 4. Requirements

All of the requirements are captured in requirements.txt. 
Run: pip install -r requirements.txt



