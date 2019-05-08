from dataset import load_data, processImg
import cv2
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

label_entries = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']


def capture_img():
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret, img = cap.read()
        cv2.rectangle(img, (450,450), (50,50), (0,255,0),0)
        crop_img = img[100:450, 100:450]
        cv2.imshow('Gesture', img)
        drawing = processImg(crop_img)
        cv2.imshow('Processed Image', drawing)
        if cv2.waitKey(33) == ord('a'):
            path = os.getcwd()+'/test.png'
            cv2.imwrite(path, drawing)
            cv2.imwrite(os.getcwd()+'/test1.png', drawing)
            cap.release()
            cv2.destroyAllWindows()
            return drawing

X_data, labels = load_data()


model = Sequential()
# Conv2D( number_of_filters , kernal_size , input_shape(add this parameter just for the input conv layer))
model.add(Conv2D(32 , (5,5) , input_shape = (120,120,1) ))
# define the activaion function for this layer
model.add(Activation('relu'))
# define the pooling for this layer
model.add(MaxPooling2D(pool_size= (2,2)))


model.add(Conv2D(64 , (3,3) ))
# define the activaion function for this layer
model.add(Activation('relu'))

model.add(Dropout(0.25))
# define the pooling for this layer
model.add(MaxPooling2D(pool_size= (2,2)))


model.add(Flatten())
model.add(Dense(1000, activation='sigmoid'))
model.add(Dense(2000, activation='sigmoid'))
model.add(Dense(1000, activation='sigmoid'))
model.add(Dense(500, activation='sigmoid'))
model.add(Dense(200, activation='relu'))
model.add(Dense(26, activation='softmax'))

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(X_data, labels, test_size=0.20, random_state=42)

model.fit(X_train, y_train, epochs=8, batch_size=50)

print('Model Trained successfully....')

predictions = model.predict(X_test)
pred = []
for x in predictions:
    pred.append(np.argmax(x))

actual = []
for x in y_test:
    actual.append(np.argmax(x))
    
print('Accuracy over the test data is: ',accuracy_score(actual, pred))
    

while True:
    cap_image = capture_img()
    cap_image = np.expand_dims(cap_image, axis=0)
    prediction = model.predict(cap_image)
    
    label_index = np.argmax(prediction)
    print('The character is: ', label_entries[label_index])
#image = capture_img()