from dataset import load_data, processImg
import cv2
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import os

label_entries = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def capture_img():
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret, img = cap.read()
        cv2.rectangle(img, (300,300), (100,100), (0,255,0),0)
        crop_img = img[100:300, 100:300]
        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        value = (35, 35)
        blurred = cv2.GaussianBlur(grey, value, 0)
        _, thresh1 = cv2.threshold(blurred, 127, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        cv2.imshow('Thresholded', thresh1)
        blur = cv2.GaussianBlur(grey,(5,5),0)
        ret,thresh1 = cv2.threshold(blur,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnt = max(contours, key = lambda x: cv2.contourArea(x))
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)
        hull = cv2.convexHull(cnt)
        drawing = np.zeros(crop_img.shape,np.uint8)
        cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)

        #cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)
        max_area=0
        hull = cv2.convexHull(cnt, returnPoints=False)
        cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)
        cv2.imshow('Gesture', img)
        all_img = np.hstack((drawing, crop_img))
        cv2.imshow('Contours', all_img)
        if cv2.waitKey(33) == ord('a'):
            drawing = cv2.resize(drawing, (100, 100))
            thresh1 = cv2.resize(thresh1, (100, 100))
            path = os.getcwd()+'/test.png'
            cv2.imwrite(path, drawing)
            cv2.imwrite(os.getcwd()+'/test1.png', thresh1)
            # test = preprocess_img(path)
            # cv2.imwrite(path, test)
            cap.release()
            cv2.destroyAllWindows()
            return drawing

X_data, labels = load_data()


model = Sequential()
# Conv2D( number_of_filters , kernal_size , input_shape(add this parameter just for the input conv layer))
model.add(Conv2D(30 , (3,3) , input_shape = (100,100,3) ))
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
model.add(Dense(2000, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(26, activation='softmax'))

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_data, labels, epochs=5, batch_size=32)

print('Model Trained successfully....')

cap_image = capture_img()
cap_image = np.expand_dims(cap_image, axis=0)
prediction = model.predict(cap_image)

label_index = np.argmax(prediction)
print('The character is: ', label_entries[label_index])
#image = capture_img()