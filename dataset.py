import os
import cv2
import numpy as np
import random

cwd = os.getcwd()+'/train'
dirs = os.listdir(cwd)

label_entries = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def processImg(img):
    cv2.rectangle(img, (300,300), (100,100), (0,255,0),0)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)
    _, thresh1 = cv2.threshold(blurred, 127, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(grey,(5,5),0)
    ret,thresh1 = cv2.threshold(blur,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key = lambda x: cv2.contourArea(x))
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 0)
    hull = cv2.convexHull(cnt)
    drawing = np.zeros(img.shape,np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)

    #cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)
    max_area=0
    hull = cv2.convexHull(cnt, returnPoints=False)
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)
    cv2.destroyAllWindows()
    #print('Shape of the image is: ', drawing.shape, thresh1.shape)
    drawing = cv2.resize(drawing, (100, 100))
    return drawing
    #return drawing, thresh1

    
def load_data():
    global label_entries
    dataset_dir = os.getcwd()+'/train'
    train_dataset = []
    labels = []
    print()
    for x in dirs:
        label_dir = dataset_dir+'/'+x
        print('Preprocessing images for: ',x)
        contents = os.listdir(label_dir)
        random.shuffle(contents)
        sampling = contents[0:50]
        for m in sampling:
            img_path = label_dir+'/'+m
            ret_img = processImg(cv2.imread(img_path))
            train_dataset.append(ret_img)
            lab = np.zeros(len(label_entries))
            lab[label_entries.index(x)] = 1
            labels.append(lab)
    return np.array(train_dataset), np.array(labels)

        
