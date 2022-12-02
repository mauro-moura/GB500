# Utils

import os
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from tensorflow.keras import backend as K

def dice_coef(y_true, y_pred, smooth = 0.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
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

def image_preprocess(train_X, train_ground, w = 320, x1 = 90, x2 = 218, size_img = 256):
    
    train_X = roi(train_X, x1, x2, w, w)
    train_ground = roi(train_ground, x1, x2, w, w)
   
    train_X = resize_img(train_X, size_img, size_img)
    train_ground = resize_img(train_ground, size_img, size_img)
    
    train_X = normalize(train_X)
    train_ground = normalize(train_ground)
    
    train_X = reshape_images(train_X, size_img,size_img)
    train_ground = reshape_images(train_ground, size_img,size_img)
    
    return train_X, train_ground

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

def plot_images(images,width, height, index, save = ""):
    curr_img = np.reshape(images[index], (width,height))
    plt.imshow(curr_img, cmap='gray')
    if save != "":
        plt.savefig(f'{save}.png')
    plt.show()
    
def unir_imagem(image1, image2, index, size_img, save = ""):
    curr_img = np.reshape(image1[index], (size_img,size_img))
    curr_img2 = np.reshape(image2[index], (size_img,size_img))
    
    new_img = curr_img + curr_img2
    
    plt.imshow(new_img, cmap='gray')
    if save != "":
        plt.savefig(f'{save}.png')
    plt.show()

def test_image():
  num = 5
  model = load_model(base_folder + 'Heart_Model_%i.h5'%(num), custom_objects = {'dice_coef_loss': dice_coef_loss, 'dice_coef' : dice_coef})

  predicao = model.predict(valid_X)
  predicao = predicao > 0.5
  #predict_vol = make_vol(predicao, index_test_y)
  predicao = np.float64(predicao)

  dice_metric.append(dice_coef(predicao, valid_ground).numpy())

  img_number = 70

  #plot_images(valid_X,size_img,size_img,img_number)
  #plot_images(predicao,size_img,size_img,img_number)
  #plot_images(valid_ground,size_img,size_img,img_number, save = True)

  unir_imagem(valid_X, valid_ground,img_number, size_img, f"{base_folder}/images/GT_img{img_number}_exec{num}")
  unir_imagem(valid_X, predicao, img_number, size_img, f"{base_folder}/images/pred_img{img_number}_exec{num}")

from keras import backend as K
import numpy as np

def Active_Contour_Loss(y_true, y_pred, debug = False):
  """
  lenth term
  """

  #          ch, n, x, y
  #          n , x, y, ch
  x = y_pred[:,1:,:,:] - y_pred[:,:-1,:,:] # horizontal and vertical directions 
  y = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:]


  delta_x = x[:,1:,:-2,:]**2
  delta_y = y[:,:-2,1:,:]**2

  delta_u = K.abs(delta_x + delta_y)

  epslon = 0.00000001
  w = 1
  lenth = w * K.mean(K.sqrt(delta_u + epslon)) # equ.(11) in the paper
  
  """
  region term
  """

  C_1 = tf.ones((256, 256))
  C_2 = tf.zeros((256, 256))
  
  region_in =   K.abs( K.mean( y_pred[:,:,:,0] * ((C_1 - y_true[:,:,:,0])**2) ) )      # equ.(12) in the paper
  region_out =  K.abs( K.mean( (1-y_pred[:,:,:,0]) * ((C_2 - y_true[:,:,:,0])**2) ) )  # equ.(12) in the paper

  lambdaP = 1 # lambda parameter could be various.
  mu = 1 # mu parameter could be various.

  eq = lenth + lambdaP * (mu * region_in + region_out)

  return eq