#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Program for Single Colorspace fused EfficientNet (SC-EffNet)"
@author: Anoop & Manjary P Gangan
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
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras import regularizers
from keras.callbacks import ModelCheckpoint   
from sklearn.metrics import classification_report  
from keras.optimizers import Adam
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

train_data_dir = 'path to train data'
validation_data_dir = 'path to validation data'
test_data_dir = 'path to test data'


###################   Scaling function  ######################
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
###########################################################


###################   Color function  ######################
# https://docs.opencv.org/master/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0a353a4b8db9040165db4dacb5bcefb6ea
# https://scikit-image.org/docs/dev/api/skimage.color.html

def colorFunction(image):

    #converted_image = cv2.cvtColor(image,cv2.COLOR_RGB2YUV) 
    color_transf_image = skimage.color.rgb2ydbdr(image)
    scaled_image = scale0to255(color_transf_image) 
    return scaled_image
###########################################################



datagen = ImageDataGenerator(preprocessing_function = colorFunction
                             )

train_generator = datagen.flow_from_directory(
        train_data_dir,
        color_mode="rgb",
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical')
validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        color_mode="rgb",
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical')
test_generator = datagen.flow_from_directory(
        test_data_dir,
        color_mode="rgb",
        target_size=(img_width, img_height),
        batch_size= 1,
        shuffle=False,
        class_mode='categorical')
train_generator.class_indices

train_samples = train_generator.samples
validation_samples = validation_generator.samples
test_samples = test_generator.samples


model_eff = EfficientNetB0(include_top = True, weights='imagenet',  input_shape = None)
model_eff.summary()

model_eff._layers.pop()
model_eff.summary()

for layer in model_eff.layers:
    layer.trainable = False

top_model = Dense(3, activation='softmax',name="prediction")(model_eff.layers[-1].output) #(top_model) 
model = Model(inputs=model_eff.input, outputs=[top_model]) 
model.summary()

keras.utils.plot_model(model, "model.png", show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001), 
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='path to save/sceffnet_color_model.h5',
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
plt.savefig('path to save/sceffnet_color_model_acc.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('path to save/sceffnet_color_model_loss.png')
plt.show()


model = load_model('path to load/sceffnet_color_model.h5')


###########################     Test case   ###########################

test_pred = model.predict_generator(test_generator,test_samples, verbose=1)
test_indx = np.argmax(test_pred,axis=1)
test_labl = (test_generator.class_indices)
test_labl = dict((v,k) for k,v in test_labl.items())
test_predictions = [test_labl[i] for i in test_indx]

test_files = test_generator.filenames
test_rslt = pd.DataFrame({"Filename":test_files,"Predictions":test_predictions})
test_rslt.to_csv("path to save predictions/sceffnet_color_model_test.csv",index=False)

print('Test Confusion Matrix')
test_target = test_generator.classes
test_predicted = test_indx
class_names=['gan', 'real', 'graphic']
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
plt.savefig('path to save confusion matrix/sceffnet_color_model_test.png')
plt.show()

print('\nTest Classification Report\n')
test_rpt = classification_report(test_target, test_predicted, target_names=class_names)
print(test_rpt)
test_acc = ((sum(test_target==test_predicted))/test_samples)*100
print('Test Accuracy = ' + str(test_acc))

test_loss= model.evaluate_generator(test_generator, test_samples)
