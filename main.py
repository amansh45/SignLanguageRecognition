from dataset import load_data, preprocess_img
import cv2
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import os

label_entries = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def capture_img():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    
    ret, frame = cam.read()
    path = os.getcwd()+'/test.png'
    cv2.imwrite(path, frame)
    cam.release()
    cv2.destroyAllWindows()
    
    test = preprocess_img(path)
    cv2.imwrite(path, test)
    return test

X_data, labels = load_data()

model = Sequential()
# Conv2D( number_of_filters , kernal_size , input_shape(add this parameter just for the input conv layer))
model.add(Conv2D(30 , (3,3) , input_shape = (100,100,1) ))
# define the activaion function for this layer
model.add(Activation('relu'))
# define the pooling for this layer
model.add(MaxPooling2D(pool_size= (2,2)))


model.add(Conv2D(30 , (3,3) ))
# define the activaion function for this layer
model.add(Activation('relu'))
# define the pooling for this layer
model.add(MaxPooling2D(pool_size= (2,2)))


model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(26, activation='softmax'))

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_data, labels, epochs=10, batch_size=32)

print('Model Trained successfully....')

cap_image = capture_img()
cap_image = np.expand_dims(cap_image, axis=0)
prediction = model.predict(cap_image)

label_index = prediction[0].tolist().index(1)
print('The character is: ', label_entries[label_index])
#image = capture_img()