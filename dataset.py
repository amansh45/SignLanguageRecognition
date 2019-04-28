import os
import cv2
import numpy as np

cwd = os.getcwd()+'/dataset'
dirs = os.listdir(cwd)

label_entries = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def preprocess_img(imgPath):
    img=cv2.imread(imgPath)

    #converting from gbr to hsv color space
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #skin color range for hsv color space 
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
    #converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    #skin color range for hsv color space 
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
    #merge skin detection (YCbCr and hsv)
    global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask=cv2.medianBlur(global_mask,3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))
    
    
    YCrCb_result = cv2.bitwise_not(YCrCb_mask)
    HSV_result = cv2.bitwise_not(HSV_mask)
    global_result=cv2.bitwise_not(global_mask)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if len(YCrCb_result.shape) == 2:
        new_image = cv2.resize(YCrCb_result, (100, 100))
        new_image = np.reshape(new_image, (new_image.shape[0],new_image.shape[1],1))
        return new_image
    else:
        return YCrCb_result
    
def load_data():
    global label_entries
    dataset_dir = os.getcwd()+'/dataset'
    train_dataset = []
    labels = []
    print()
    for x in dirs:
        label_dir = dataset_dir+'/'+x
        print('Preprocessing images for: ',x)
        for m in os.listdir(label_dir):
            img_path = label_dir+'/'+m
            train_dataset.append(preprocess_img(img_path))
            lab = np.zeros(len(label_entries))
            lab[label_entries.index(x)] = 1
            labels.append(lab)
    return np.array(train_dataset), np.array(labels)

        