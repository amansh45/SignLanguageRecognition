import os
import cv2
import numpy as np
import random

cwd = os.getcwd()+'/train'
dirs = os.listdir(cwd)

label_entries = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

    
def processImg(frame):
    # converting BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # define range of red color in HSV
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])
    
    # create a red HSV colour boundary and
    # threshold HSV image
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    edges = cv2.Canny(frame,100,200)
    edges = cv2.resize(edges, (120, 120))
    edges = np.reshape(edges,(edges.shape[0], edges.shape[1], 1))
    return edges

    
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
            k = cv2.waitKey(5) & 0xFF
            spath = os.getcwd()+'/test/'+m
            cv2.imwrite(spath, ret_img)
            train_dataset.append(ret_img)
            lab = np.zeros(len(label_entries))
            lab[label_entries.index(x)] = 1
            labels.append(lab)
    return np.array(train_dataset), np.array(labels)

        
