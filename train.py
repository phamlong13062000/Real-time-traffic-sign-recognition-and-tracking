import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random

from tensorflow.python.keras.layers.core import Activation

dataPath = 'D:/Machine_Learning/DATA/mydata3'

def loadDataSet(dataPath):
    data = []
    imageSize = 48
    for cat in os.listdir(dataPath):
        label = int(cat[:2])
        for i in os.listdir(dataPath+'/'+cat):
            imagePath = dataPath+'/'+cat+'/'+i
            try:
                image = cv2.imread(imagePath,0) 
                image = cv2.resize(image,(imageSize,imageSize))
                data.append([image,label])
            except:
                pass
    
    random.shuffle(data)
    
    X = [] #image
    Y = [] #labels
    for img,l in data: 
        X.append(img)
        Y.append(l)
    X = np.array(X)
    Y = np.array(Y).reshape(len(Y),1)
    return X,Y

images,labels=loadDataSet(dataPath)

for i in range(10):
    plt.imshow(images[i],cmap='gray')
    plt.show()
    print(labels[i])

mu=np.mean(images)
std=np.std(images)
print("The Mean=",mu)
print("The Standard Deviation=",std)

print(images.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(images,labels,test_size=0.2,random_state=0)
print("Training =",x_train.shape[0])
print("Testing =",x_test.shape[0])


x_trainNorm = (x_train - mu)/std
x_testNorm  = (x_test - mu)/std
x_trainNorm = x_trainNorm.reshape(2928, 48, 48, 1)
x_testNorm = x_testNorm.reshape(733, 48, 48, 1)

def preprocessingImage(image=None,imageSize=28,mu=102.23982103497072,std=72.11947698025735):
    try:
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    except:
        pass
    image = cv2.resize(image,(imageSize,imageSize))
    image = (image - mu) / std
    image = image.reshape(1,imageSize,imageSize,1)
    return image

import tensorflow as tf
from tensorflow import keras

class myCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy')>0.997):
            self.model.stop_training=True
            

model = keras.Sequential([
    keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(48,48,1)),
    keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Dropout(0.25),

    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(9,tf.nn.softmax)
])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
model.summary()

model.fit(x_trainNorm,
          y_train,
          epochs = 15,
          callbacks = [myCallBack()])

acc,loss =model.evaluate(x_testNorm,
               y_test,
               verbose = 0)
print(acc,loss)

model.save('D:/Machine_Learning/FINALL/model/mymodel_o13')