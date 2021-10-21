#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
InceptionResnet with transfer learning for classifying GAN, Graphics, Real 
Train and Test Module in the same .py file

@authors: Anoop & Manjary P Gangan 
"""

import keras
print(keras.__version__)

import tensorflow
print(tensorflow.__version__)

import mlxtend                                                        
print(mlxtend.__version__) 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint   
from sklearn.metrics import classification_report  
from keras.optimizers import Adam
from keras.models import load_model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

img_width, img_height = 299, 299
batch_size = 256
epochs = 100                                
train_samples = 7200
validation_samples = 2400
test_samples = 2400

train_data_dir = 'path to train data'
validation_data_dir = 'path to validation data'
test_data_dir = 'path to test data'

datagen = ImageDataGenerator(rescale = 1./255)

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

model_incres = keras.applications.InceptionResNetV2(include_top = True, weights='imagenet',  input_shape = None)
model_incres.summary()

model_incres._layers.pop()
model_incres.summary()

for layer in model_incres.layers:
    layer.trainable = False

top_model = Dense(3, activation='softmax',name="prediction")(model_incres.layers[-1].output) 
model = Model(inputs=model_incres.input, outputs=[top_model]) 
model.summary()

keras.utils.plot_model(model, "model.png", show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.01), 
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='path to model/model_name.h5',
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
plt.savefig('path to accuracy plot/acc.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('path to loss plot/loss.png')
plt.show()


###########################     Test case   ###########################

model = load_model('path to model/model_name.h5')

test_pred = model.predict_generator(test_generator,test_samples, verbose=1)
test_indx = np.argmax(test_pred,axis=1)
test_labl = (test_generator.class_indices)
test_labl = dict((v,k) for k,v in test_labl.items())
test_predictions = [test_labl[i] for i in test_indx]

test_files = test_generator.filenames
test_rslt = pd.DataFrame({"Filename":test_files,"Predictions":test_predictions})
test_rslt.to_csv("path to save model predictions/model_prediction.csv",index=False)

print('Test Confusion Matrix')
test_target = test_generator.classes
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
plt.savefig('path to save confusion matrix/conf_matrix.png')
plt.show()

print('\nTest Classification Report\n')
test_rpt = classification_report(test_target, test_predicted, target_names=class_names)
print(test_rpt)
test_acc = ((sum(test_target==test_predicted))/test_samples)*100
print('Test Accuracy = ' + str(test_acc))

test_loss= model.evaluate_generator(test_generator, test_samples)
test_loss
