#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1)import all required modules
import tensorflow as tf
import tensorflow_datasets as tfds
import math

import matplotlib.pyplot as plt
import numpy as np

import os
import zipfile

import random

import cv2

#from IPython import display

from keras.layers import Conv2D, UpSampling2D

#import PIL.Image, PIL.ImageFont, PIL.ImageDraw
from matplotlib import image

#from sklearn.model_selection import train_test_split
#from tensorflow.keras import layers, losses

print("Tensorflow version " + tf.__version__)

#import PIL
#print('Pillow Version:', PIL.__version__)


# In[4]:


#load required dataset
local_zip = './colored.zip'
local_dir = './training'

zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall(local_dir)
zip_ref.close()


# In[32]:


#3)check if there is improper image formate 
#no need to run again
#https://www.pythonfixing.com/2021/12/fixed-tensorflow-keras-error-unknown.html
#import os
#import cv2

def check_images( s_dir, ext_list):
    bad_images=[]
    bad_ext=[]
    s_list= os.listdir(s_dir)
    for klass in s_list:
        klass_path=os.path.join (s_dir, klass)
        print ('processing class directory ', klass)
        if os.path.isdir(klass_path):
            file_list=os.listdir(klass_path)
            for f in file_list:               
                f_path=os.path.join (klass_path,f)
                index=f.rfind('.')
                ext=f[index+1:].lower()
                if ext not in ext_list:
                    print('file ', f_path, ' has an invalid extension ', ext)
                    bad_ext.append(f_path)
                if os.path.isfile(f_path):
                    try:
                        img=cv2.imread(f_path)
                        shape=img.shape
                    except:
                        print('file ', f_path, ' is not a valid image file')
                        bad_images.append(f_path)
                else:
                    print('*** fatal error, you a sub directory ', f, ' in class directory ', klass)
        else:
            print ('*** WARNING*** you have files in ', s_dir, ' it should only contain sub directories')
    return bad_images, bad_ext

source_dir =r'./training'
good_exts=['jpg'] # list of acceptable extensions
bad_file_list, bad_ext_list=check_images(source_dir, good_exts)
if len(bad_file_list) !=0:
    print('improper image files are listed below')
    for i in range (len(bad_file_list)):
        print (bad_file_list[i])
else:
    print(' no improper image files were found')


# In[6]:


#removing improper images
os.remove('./training/colored/cl_2_14112_284659IM000001_052_S.JPG')


# In[2]:


img_data_dir = os.path.join('./training/colored') #train dir
img_names_list = os.listdir(img_data_dir) #img name list

#generating list containing the image paths
img_paths_list = [os.path.join(img_data_dir, fname) for fname in img_names_list]
print(img_paths_list[0])


# In[3]:


# split the paths list into to training (80%) and testing sets(20%).
paths_len = len(img_paths_list)
train_paths_len = int(paths_len * 0.8)

train_paths_list = img_paths_list[:train_paths_len]
test_paths_list = img_paths_list[train_paths_len:]


# In[4]:


#check
print(len(train_paths_list))
print(len(test_paths_list))
print((paths_len))


# In[5]:


IMG_HEIGHT=480 #491 #896
IMG_WIDTH=480 #493 #896

def prepro_resize(input_img):
    oimg=image.imread(input_img)
    return cv2.resize(oimg, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)


# In[7]:


x_train=train_paths_list
print(len(x_train))


# In[8]:



#x_train_ = [prepro_resize(x_train[i]) for i in range(len(x_train))]
#x_train_ = np.array(x_train_)
#x_train_=x_train_.astype('float32')/255

#taking 1000 sample as there is a memory problem woring on 4k samples
x_train_ = [(prepro_resize(x_train[i])).astype('float32')/255.0 for i in range(len(x_train))] #(1000) 

x_test=test_paths_list
x_test_ = [(prepro_resize(x_test[i])).astype('float32')/255.0 for i in range(len(x_test))]


# In[39]:


plt.imshow(x_train_[0])
plt.show()


# In[40]:


#plot diff images

plt.figure(figsize=(8, 8))
plt.subplot(2,2,1)
plt.imshow(x_train_[10])
plt.subplot(2,2,2)
plt.imshow(x_train_[1])
plt.subplot(2,2,3)
plt.imshow(x_train_[15])
plt.subplot(2,2,4)
plt.imshow(x_train_[150])
plt.show()


# In[20]:


#x_train_set = np.array(x_train_)
#print(x_train_set.shape)


# In[9]:


#x_test_small_set =np.array(x_train_[1:301])
x_train_small_set =np.array(x_train_[1:1001])
print(x_train_small_set.shape)
#x_train_set = np.array(x_train_)
#print(x_train_set.shape)


# In[16]:


print(len(x_train_))
#print(len(x_train_set))


# In[42]:


print(len(x_train_small_set))


# In[ ]:


#mo7awalat to be considered for AE
#1)layers,kernel
#2)Dropout(x)
#3)change loss function used
#4)Conv2D instead of Conv2dTranspose
#5)try diffrent encoder design (DAE,VAE)


# In[16]:


#Encoder
def encoder(inputs):
    #conv_0 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3),activation='relu', padding='same', name='enc_conv_0')(inputs)
    #max_pool_0 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv_0)
    
    conv_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3),activation='relu', padding='same', name='enc_conv_1')(inputs)#max_pool_0
    max_pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv_1)
  
    conv_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', name='enc_conv_2')(max_pool_1)
    max_pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv_2)
    
    conv_3 = tf.keras.layers.Conv2D(filters=255, kernel_size=(3,3), activation='relu', padding='same', name='enc_conv_3')(max_pool_2)
    max_pool_3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv_3)
    
    conv_4 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='enc_conv_4')(max_pool_3)
    max_pool_4 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv_4)
    
    conv_5 = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3,3), activation='relu', padding='same', name='enc_conv_5')(max_pool_4)
    max_pool_5 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv_5)
    
    return max_pool_5
  


# In[17]:


#bottle_neck
def bottle_neck(inputs):
    bottle_neck = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3,3),activation='relu', padding='same', name='bottle_neck')(inputs)
    
    encoder_visualization = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(3,3),activation='relu', padding='same', name='dec_conv1')(bottle_neck)  
    encoder_visualization = tf.keras.layers.UpSampling2D(size = (16,16))(encoder_visualization)
    encoder_visualization = tf.keras.layers.Resizing(224,224)(encoder_visualization)
    
    return bottle_neck, encoder_visualization


# In[18]:


#Decoder
def decoder(inputs):
    conv_1 = tf.keras.layers.Conv2DTranspose(filters=1024, kernel_size=(3,3),activation='relu', padding='same', name='dec_conv1')(inputs)
    up_sample_1 = tf.keras.layers.UpSampling2D(size=(2,2))(conv_1)
    
    conv_2 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=(3,3),activation='relu', padding='same', name='dec_conv2')(up_sample_1)
    up_sample_2 = tf.keras.layers.UpSampling2D(size=(2,2))(conv_2)
    
    conv_3 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(3,3),activation='relu', padding='same', name='dec_conv3')(up_sample_2)
    up_sample_3 = tf.keras.layers.UpSampling2D(size=(2,2))(conv_3)
    
    conv_4 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3,3),activation='relu', padding='same', name='dec_conv4')(up_sample_3)
    up_sample_4 = tf.keras.layers.UpSampling2D(size=(2,2))(conv_4)
    
    conv_5 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3,3),activation='relu', padding='same', name='dec_conv5')(up_sample_4)
    up_sample_5 = tf.keras.layers.UpSampling2D(size=(2,2))(conv_5)
    
    #conv_ = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3,3),activation='relu', padding='same', name='dec_conv5')(up_sample_5)
    #up_sample_ = tf.keras.layers.UpSampling2D(size=(2,2))(conv_)
    
    conv_6 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(3,3),activation='sigmoid',padding='same', name='dec_conv6')(up_sample_5)#up_sample_
    
    return conv_6


# In[19]:


#CAE block

def convolutional_auto_encoder():
    inputs = tf.keras.layers.Input(shape=(480,480,3))#(896,896,3))#(491, 493, 3,)
    #x = tf.keras.layers.Flatten(inputs)
    
    encoder_output = encoder(inputs)
    
    bottleneck_output, encoder_visualization = bottle_neck(encoder_output)
    
    decoder_output = decoder(bottleneck_output)
    print(encoder_visualization.shape)
    
    #op=tf.keras.layers.Reshape((491,493,3))
    
    model = tf.keras.Model(inputs =inputs, outputs=decoder_output)
    
    encoder_model = tf.keras.Model(inputs=inputs, outputs=encoder_visualization)
    
    # add the KL loss
    ##loss = kl_reconstruction_loss(inputs, decoder_output, mu, sigma)
    ##model.add_loss(loss)
    
    return model, encoder_model


# In[20]:


#Model
import tensorflow as tf
model,encoder_model=convolutional_auto_encoder()


# In[33]:


#model
model.summary()


# In[10]:


#plot model arch
#tf.keras.utils.plot_model(model,show_shapes=True)


# In[21]:


#incorrect loss method check for another suitable loss methods to use#####
def JSD_Tensor_loss(P, Q):
    M=tf.math.divide((tf.math.add(P,Q)),2)
    
    D1=tf.math.multiply(P,(tf.math.log(P)))
    D2=tf.math.multiply(Q,(tf.math.log(Q)))
    
    JSD=tf.math.divide((tf.add(D1,D2)),2)
    JSD=tf.math.reduce_sum(JSD)
    
    return JSD

def KLD_loss(inputs, outputs, mu, sigma):
    kl_loss = 1 + sigma - tf.square(mu) - tf.math.exp(sigma)
    return tf.reduce_mean(kl_loss) * -0.5

def JSD_loss(P, Q):
    P=P.numpy() 
    Q=Q.numpy() 
    _P = P / np.linalg.norm(P, ord=1)
    _Q = Q / np.linalg.norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (math.entropy(_P, _M) + math.entropy(_Q, _M))


# calculate the kl divergence
def kl_divergence(p, q):
    return sum(p[i] * math.log2(p[i]/q[i]) for i in range(len(p)))
 
# calculate the js divergence
def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


# In[22]:


#Train

#loss = tf.keras.losses.KLDivergence() 

#loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)XXXXX

#loss='sparse_categorical_crossentropy'
#loss='JSD_loss'XXX
#loss=tf.keras.losses.MeanSquaredError()

#loss=js_divergence() #check for better loss
#loss=JSD_Tensor_loss
#loss=tf.keras.losses.Huber()
#loss=tf.keras.losses.LogCosh()

optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
#optimizer='adam'#check for better optimizer
#optimizer='sgd'


# In[23]:


#needed callbacks and checkpoints
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') >= 0.60): #changing the value
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()


# In[24]:


model.compile(optimizer=optimizer, 
              loss=tf.keras.losses.LogCosh(), #loss,
              metrics=['accuracy']) #check for suitable metrics other than acc


# In[25]:


print(x_train_small_set.shape)


# In[26]:


train_dataset=x_train_small_set[:501]
print(train_dataset.shape)


# In[27]:


# fit the autoencoder model 

# history=model.fit(x_train_, x_train_, epochs=100) #Memory Error

history=model.fit(train_dataset, train_dataset, epochs=10,callbacks=[callbacks])#,callbacks=[callbacks,tf.keras.callbacks.TensorBoard(log_dir='./logs')])


# In[28]:


print(history.history.keys())

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[43]:


x_test=test_paths_list
x_test_ = [(prepro_resize(x_test[i])).astype('float32')/255.0 for i in range(len(x_test))]

x_test_small_set =np.array(x_train_[:501])


# In[54]:


#to display results

IMG_HEIGHT=480 #491
IMG_WIDTH=480 #493
DEPTH=3

def display_one_row(disp_images, offset, shape=(IMG_HEIGHT, IMG_WIDTH, DEPTH)):
    '''Display sample outputs in one row.'''
    for idx, test_image in enumerate(disp_images):
        plt.subplot(3, 10, offset + idx + 1)
        plt.xticks([])
        plt.yticks([])
        test_image = np.reshape(test_image, shape)
        plt.imshow(test_image, cmap='gray')



def display_results(disp_input_images, disp_encoded, disp_predicted, enc_shape=(224,224,3)):
    '''Displays the input, encoded, and decoded output values.'''
    plt.figure(figsize=(15, 5))
    display_one_row(disp_input_images, 0, shape=(IMG_HEIGHT, IMG_WIDTH, DEPTH))
     
    display_one_row(disp_encoded, 10, shape=enc_shape)
    
    display_one_row(disp_predicted, 20, shape=(IMG_HEIGHT, IMG_WIDTH, 1))
   


# In[46]:


test_dataset=x_test_small_set

#encoder output
encoded_predicted = encoder_model.predict(test_dataset)


# test prediction
simple_predicted = model.predict(test_dataset)


# In[49]:


import time
BATCH_SIZE = 128
output_samples=x_test_small_set

# pick 10 random numbers 
np.random.seed(int(time.time())+random.randint(0,10))
idxs = np.random.choice(BATCH_SIZE, size=10)

# display 10 samples, encodings and decoded values!
display_results(output_samples[idxs], encoded_predicted[idxs], simple_predicted[idxs])


# In[55]:


# display the 10 samples, encodings and decoded values!
display_results(output_samples[idxs], encoded_predicted[idxs], simple_predicted[idxs])


# In[ ]:




