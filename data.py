import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os

REQ_W = 224
REQ_H = 224
TRAIN_IMAGES_INPUT_PATH = 'road_segmentation_ideal/training/input/'
TRAIN_IMAGES_OUTPUT_PATH = 'road_segmentation_ideal/training/output/'

# function to load the dataset
def load_dataset():
    # loads the dataset files into a list
    train_images_inp_list = os.listdir(TRAIN_IMAGES_INPUT_PATH)
    train_images_out_list = os.listdir(TRAIN_IMAGES_OUTPUT_PATH)
    inp_img_path = []
    out_img_path = []
    for p in train_images_inp_list:
        inp_img_path.append(os.path.split(p)[1])
    for p in train_images_out_list:
        out_img_path.append(os.path.split(p)[1])
    train_images_path = list(set(inp_img_path).intersection(out_img_path))
    # show first 4 images as trial
    # uncomment these to show the images while loading the dataset
    # for i in range(2):
    #     img_inp = input_encoding(TRAIN_IMAGES_INPUT_PATH+train_images_path[i])
    #     img_out = output_encoding(TRAIN_IMAGES_OUTPUT_PATH+train_images_path[i])
    #     display(img_inp,img_out,train_images_path[i])
    # print("Number of training samples:"+str(len(train_images_path)))
    return train_images_path

# function to read and preprocess the input image of network
def input_encoding(file_name):
    print(file_name)
    image = cv2.imread(filename=file_name)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    h,w,c = image.shape
    image = cv2.resize(image,(REQ_W,REQ_H))
    image = image/255.0
    return image

# function to read and preprocess the output image of network
def output_encoding(file_name):
    print(file_name)
    image = cv2.imread(filename=file_name)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    h,w = image.shape
    image = cv2.resize(image,(REQ_W,REQ_H))
    image = image/255.0
    return image

def display(img_inp,img_out,img_name):
    fig = plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.title(img_name)
    plt.imshow(img_inp)

    plt.subplot(1,2,2)
    plt.title(img_name)
    plt.imshow(img_out,cmap=plt.get_cmap('gray'))
    plt.show()