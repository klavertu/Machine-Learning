# Kelly Lavertu
# CS 7267 Machine Learning Project
# Fall 2022

# imports
from time import time

import keras
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import Sequential
import tensorflow as tf
import glob

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tensorflow_datasets.image_classification.colorectal_histology_test import num_classes
import os

# creating paths to galaxy and star image folders and another testing set

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

galaxy_path = glob.glob(
    'C:/Users/Kelly/Google Drive/School/Masters/Fall2022/CS7267_MachineLearning/project/archive/CutoutFiles/galaxy/*')
star_path = glob.glob(
    'C:/Users/Kelly/Google Drive/School/Masters/Fall2022/CS7267_MachineLearning/project/archive/CutoutFiles/star/*')
new_train_path = glob.glob(
    'C:/Users/Kelly/Google Drive/School/Masters/Fall2022/CS7267_MachineLearning/project/archive/CutoutFiles/preprocessed_images/*')

# loading images into the appropriate dataset to label them properly; 0 - galaxy 1 - star

data = []
labels = []

for x in galaxy_path:
    image = cv2.imread(x, 1)
    image = np.array(image)
    data.append(image)
    labels.append(0)
for x in star_path:
    image = cv2.imread(x, 1)
    image = np.array(image)
    data.append(image)
    labels.append(1)

# putting the data and labels into an array, then into a dataframe to better handle processing
data, labels = np.array(data), np.array(labels)
dataset = pd.DataFrame(list(zip(data, labels)), columns=['images', 'labels'])

# splitting the testing and training set - 75%/25%
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25,
                                                    random_state=42, shuffle=True, stratify=dataset['labels'].values)
# shaping and normalizing the data
x_train.shape, x_test.shape

x_train = x_train / 255.0
x_test = x_test / 255.0

# sklearn expects i/p to be 2d array-model.fit(x_train,y_train)=>reshape to 2d array
nsamples, nx, ny, nrgb = x_train.shape
x_train2 = x_train.reshape((nsamples, nx * ny * nrgb))

# so,eventually,model.predict() should also be a 2d input
nsamples, nx, ny, nrgb = x_test.shape
x_test2 = x_test.reshape((nsamples, nx * ny * nrgb))

print('xtrain shape\n', x_train2.shape)
print('xtest shape\n', x_test2.shape)

# start of decision tree model
dtc = DecisionTreeClassifier(max_depth=6)

dtc.fit(x_train2, y_train)
print('training complete')
y_pred_dtc = dtc.predict(x_test2)
y_pred_dtc

# print accuracy score, classification report, and confusion matrix for decision tree

accuracy_score(y_pred_dtc, y_test)
print(classification_report(y_pred_dtc, y_test))

cmd_obj = ConfusionMatrixDisplay(confusion_matrix(y_pred_dtc, y_test), display_labels=['Galaxy', 'Star'])
cmd_obj.plot()
plt.show()

cmd_obj.ax_.set(
    title='Predicted vs. Actual Celestial Objects',
    xlabel='Predicted Celestial Objects',
    ylabel='Actual Celestial Objects')

# start of CNN
# define our model --
# adds the convolutional and ReLU layers, as well as the pooling and dense layers necessary for the CNN
t0 = time()
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation=Activation(tf.nn.softmax)))

# when compiling the model, the sparse categorical cross entropy loss function is used
# used the Adam optimizer which is an extension of the Stochastic Gradient Descent, epochs set to 10 to improve accuracy

model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs =10, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])
print("\nTime Elapsed: ", (time() - t0) / 60, ' minutes')
