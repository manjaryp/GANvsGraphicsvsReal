"""
Program for Multi-Colorspace fused EfficientNet (MC-EffNet-2)
@author: Manjary & Anoop 
"""

import keras
import tensorflow
import mlxtend                                                        
import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, concatenate, Concatenate#, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint   
from sklearn.metrics import classification_report  
from tensorflow.keras.optimizers import Adam
from skimage import color
from tensorflow.keras.applications.efficientnet import *


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

img_width, img_height = 224, 224
batch_size = 256
epochs = 100                                
train_samples = 7200
validation_samples = 2400
test_samples = 2400

train_data_dir      = 'path to train data'
validation_data_dir = 'path to validation data'
test_data_dir       = 'path to test data'


##################  Rescaling function ########################################

def scale0to255(image):
    converted_image = image
    min_1 = np.min(converted_image[:,:,0])
    max_1 = np.max(converted_image[:,:,0])
    converted_image[:,:,0] = np.round(((converted_image[:,:,0] - min_1) / (max_1 - min_1)) * 255)
   
    min_2 = np.min(converted_image[:,:,1])
    max_2 = np.max(converted_image[:,:,1])
    converted_image[:,:,1] = np.round(((converted_image[:,:,1] - min_2) / (max_2 - min_2)) * 255)
    
    min_3 = np.min(converted_image[:,:,2])
    max_3 = np.max(converted_image[:,:,2])
    converted_image[:,:,2] = np.round(((converted_image[:,:,2] - min_3) / (max_3 - min_3)) * 255)
    return converted_image 

##################  Laplacian of gaussian pre-processing function #############

def log(image):
    gaus_image = cv2.GaussianBlur(image,(3,3),0)
    laplacian_image = cv2.Laplacian(np.uint8(gaus_image), cv2.CV_64F)
    sharp_image = np.uint8(image + laplacian_image)
    return sharp_image 

##################  LCH color transformation function #########################

def lch_colorFunction(image):
    log_image = log(image)
    lab_image = skimage.color.rgb2lab(log_image)
    lch_image = skimage.color.lab2lch(lab_image)
    scale_lch_image = scale0to255(lch_image)
    return scale_lch_image

##################  HSV color transformation function #########################

def hsv_colorFunction(image):
    log_image = log(image)
    hsv_image  = skimage.color.rgb2hsv(log_image)
    scale_hsv_image = scale0to255(hsv_image)
    np.nan_to_num(scale_hsv_image, copy=False, nan=0.0, posinf=None, neginf=None)
    return scale_hsv_image 

####################### Data preparation ######################################

datagen_rgb = ImageDataGenerator()
datagen_lch = ImageDataGenerator(preprocessing_function = lch_colorFunction)
datagen_hsv = ImageDataGenerator(preprocessing_function = hsv_colorFunction)

def myGenerator (gen1, gen2, gen3):#
  while True:
    xy1 = gen1.next()
    xy2 = gen2.next()
    xy3 = gen3.next()
    yield ([xy1[0], xy2[0], xy3[0]], xy1[1]) #

train_generator_rgb = datagen_rgb.flow_from_directory(
                      train_data_dir,
                      color_mode="rgb",
                      target_size=(img_width, img_height),
                      batch_size=batch_size,
                      shuffle=False,
                      class_mode='categorical')

train_generator_lch = datagen_lch.flow_from_directory(
                      train_data_dir,
                      color_mode="rgb",
                      target_size=(img_width, img_height),
                      batch_size=batch_size,
                      shuffle=False,
                      class_mode='categorical')

train_generator_hsv = datagen_hsv.flow_from_directory(
                      train_data_dir,
                      color_mode="rgb",
                      target_size=(img_width, img_height),
                      batch_size=batch_size,
                      shuffle=False,
                      class_mode='categorical')

train_generator = myGenerator(train_generator_rgb, train_generator_lch, train_generator_hsv)#

validation_generator_rgb = datagen_rgb.flow_from_directory(
                           validation_data_dir,
                           color_mode="rgb",
                           target_size=(img_width, img_height),
                           batch_size=batch_size,
                           shuffle=False,
                           class_mode='categorical')

validation_generator_lch = datagen_lch.flow_from_directory(
                           validation_data_dir,
                           color_mode="rgb",
                           target_size=(img_width, img_height),
                           batch_size=batch_size,
                           shuffle=False,
                           class_mode='categorical')

validation_generator_hsv = datagen_hsv.flow_from_directory(
                           validation_data_dir,
                           color_mode="rgb",
                           target_size=(img_width, img_height),
                           batch_size=batch_size,
                           shuffle=False,
                           class_mode='categorical')

validation_generator = myGenerator(validation_generator_rgb, validation_generator_lch, validation_generator_hsv)#

test_generator_rgb = datagen_rgb.flow_from_directory(
                     test_data_dir,
                     color_mode="rgb",
                     target_size=(img_width, img_height),
                     batch_size= 1,
                     shuffle=False,
                     class_mode='categorical')

test_generator_lch = datagen_lch.flow_from_directory(
                     test_data_dir,
                     color_mode="rgb",
                     target_size=(img_width, img_height),
                     batch_size= 1,
                     shuffle=False,
                     class_mode='categorical')

test_generator_hsv = datagen_hsv.flow_from_directory(
                     test_data_dir,
                     color_mode="rgb",
                     target_size=(img_width, img_height),
                     batch_size= 1,
                     shuffle=False,
                     class_mode='categorical')

test_generator = myGenerator(test_generator_rgb, test_generator_lch, test_generator_hsv)#


####################### Model construction ####################################

model_eff_rgb = EfficientNetB0(include_top = True, weights='imagenet',  input_shape = None)
model_eff_rgb._layers.pop()
for layer in model_eff_rgb.layers:
    layer.trainable = False
for i, layer in enumerate(model_eff_rgb.layers):
  layer._name = 'eff_rgb_'+str(i)
model_eff_rgb.summary()

model_eff_lch = EfficientNetB0(include_top = True, weights='imagenet',  input_shape = None)
model_eff_lch._layers.pop()
for layer in model_eff_lch.layers:
    layer._trainable = False
for i, layer in enumerate(model_eff_lch.layers):
  layer._name = 'eff_lch_'+str(i)
model_eff_lch.summary()

model_eff_hsv = EfficientNetB0(include_top = True, weights='imagenet',  input_shape = None)
model_eff_hsv._layers.pop()
for layer in model_eff_hsv.layers:
    layer._trainable = False
for i, layer in enumerate(model_eff_hsv.layers):
  layer._name = 'eff_hsv_'+str(i)
model_eff_hsv.summary()

fused_1 = concatenate([model_eff_rgb.layers[-1].output, model_eff_lch.layers[-1].output, model_eff_hsv.layers[-1].output])

top_model = Dense(3, activation='softmax',name="prediction")(fused_1) 
model = Model(inputs=[model_eff_rgb.input, model_eff_lch.input, model_eff_hsv.input], outputs=[top_model]) 
model.summary()

keras.utils.plot_model(model, "path to save/model _architecture.png", show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001), 
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='path to save/mceffnet_model.h5',
                               verbose=1,
                               save_best_only=True)

history = model.fit_generator(train_generator,
                              steps_per_epoch=train_samples / batch_size,
                              epochs=epochs, callbacks=[checkpointer],
                              validation_data=validation_generator,
                              validation_steps=validation_samples / batch_size)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('path to save/model_acc.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('path to save/model_loss.png')
plt.show()


model = load_model('path to load/mceffnet_model.h5')
model.summary()

###########################     Test case   ###########################

test_pred = model.predict_generator(test_generator,test_samples, verbose=1)
test_indx = np.argmax(test_pred,axis=1)
test_labl = (test_generator_rgb.class_indices)
test_labl = dict((v,k) for k,v in test_labl.items())
test_predictions = [test_labl[i] for i in test_indx]

test_files = test_generator_rgb.filenames
test_rslt = pd.DataFrame({"Filename":test_files,"Predictions":test_predictions})
test_rslt.to_csv("path to save/model_predictions.csv",index=False)

print('Test Confusion Matrix')
test_target = test_generator_rgb.classes
test_predicted = test_indx
class_names=['gan', 'graphic', 'real']
cm = confusion_matrix(y_target=test_target, 
                      y_predicted=test_predicted, 
                      binary=False)
fig, ax = plot_confusion_matrix(conf_mat=cm,
                                show_normed=True,
                                cmap="YlGnBu",
                                colorbar=True,
                                class_names=class_names)
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names,rotation=0)
plt.yticks(tick_marks, class_names)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('top')
plt.savefig('path to save/confusion_matrix.png')
plt.show()

print('\nTest Classification Report\n')
test_rpt = classification_report(test_target, test_predicted, target_names=class_names)
print(test_rpt)
test_acc = ((sum(test_target==test_predicted))/test_samples)*100
print('Test Accuracy = ' + str(test_acc))

test_loss= model.evaluate_generator(test_generator, test_samples)
