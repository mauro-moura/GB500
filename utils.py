# Utils

import os
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def dice_coef(y_true, y_pred, smooth = 0.):
    var_type = tf.float64
    y_true_f = tf.cast(y_true, var_type)
    y_true_f = K.flatten(y_true_f)

    y_pred_f = y_pred > 0.5
    y_pred_f = tf.cast(y_pred, var_type)
    y_pred_f = K.flatten(y_pred_f)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

def roi(img, x1, x2, w, h):
    imagem = []
    for i in range(len(img)):
        curr_img = np.reshape(img[i], (w,h))
        imagem.append(curr_img[x1:x2,x1:x2])
    return imagem

def resize_img(img, width, heigh):
    resized = []
    for i in range(len(img)):
        resized.append(cv2.resize(img[i], (width, heigh)))
    return resized

def reshape_images(images, width, height):
    images = np.asarray(images)
    images = images.reshape(-1, width, height, 1)
    return images

def normalize(images):
    m = np.max(images)
    mi = np.min(images)
    images = (images - mi) / (m - mi)
    return images

def image_preprocess(train_X, valid_X, train_ground, valid_ground, w = 320, x1 = 90, x2 = 218, size_img = 256):
    
    train_X, valid_X = roi(train_X, x1, x2, w, w), roi(valid_X, x1, x2, w, w)
    train_ground, valid_ground = roi(train_ground, x1, x2, w, w), roi(valid_ground, x1, x2, w, w)
   
    train_X, valid_X = resize_img(train_X, size_img, size_img), resize_img(valid_X, size_img, size_img)
    train_ground, valid_ground = resize_img(train_ground, size_img, size_img), resize_img(valid_ground, size_img, size_img)
    
    train_X, valid_X = normalize(train_X), normalize(valid_X)
    train_ground, valid_ground = normalize(train_ground), normalize(valid_ground)
    
    train_X, valid_X = reshape_images(train_X, size_img,size_img), reshape_images(valid_X, size_img,size_img)
    train_ground, valid_ground = reshape_images(train_ground, size_img,size_img), reshape_images(valid_ground, size_img,size_img)
    
    return train_X, valid_X, train_ground, valid_ground

def get_images(image,hd = None,hu = None):
    images = []
    #index = []
    for f in range(len(image)):
        ai = []
        a = nib.load(image[f])
        a = a.get_data()
        if (hd.any() != None and hu.any() != None):
            a = a[:,:,hd[f]:hu[f]]
        for i in range(a.shape[2]):
            ai.append((a[:,:,i]))
        images += ai
        #index.append(len(ai))
    """
    a = []
    for X in train_X:
      a += X
    """
    return images

def unir_imagem(image1, image2, index, size_img, save = ""):
    curr_img = np.reshape(image1[index], (size_img,size_img))
    curr_img2 = np.reshape(image2[index], (size_img,size_img))
    
    new_img = curr_img + curr_img2
    
    plt.imshow(new_img, cmap='gray')
    if save != "":
        plt.savefig(f'{save}.png')
    plt.show()

def plot_images(images,width, height, index, save = ""):
    curr_img = np.reshape(images[index], (width,height))
    plt.imshow(curr_img, cmap='gray')
    if save != "":
        plt.savefig(f'{save}.png')
    plt.show()

def create_folder(dirName):
    # Create target Directory if don't exist
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Diretorio " , dirName ,  " Criado ")
    else:    
        print("Diretorio " , dirName ,  " ja existe")
