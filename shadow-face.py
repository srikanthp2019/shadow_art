import cv2
from keras.models import load_model
import numpy as np
import os
import sys
import logging as log
import datetime as dt
from time import sleep
import subprocess

model = load_model('shadow.h5')

cascPath = "haarcascade_frontalface_default.xml"  # for face detection

if not os.path.exists(cascPath):
    subprocess.call(['./download_filters.sh'])
else:
    print('Filters already exist!')

def put_mask(mst,fc,x,y,w,h):
    
    face_width = w
    face_height = h

    mst_width = int(face_width)-1
    mst_height = int(face_height)-1

    mst = cv2.resize(mst,(mst_width,mst_height))

    for i in range(0,mst_height):
        for j in range(0,mst_width):
            for k in range(3):
                if mst[i][j][k] <235:
                    fc[y+i][x+j][k] = mst[i][j][k]
    return fc

def main():
    images = get_images()
    cap = cv2.VideoCapture(0)
    faceCascade = cv2.CascadeClassifier(cascPath)
    x, y, w, h = 700, 50, 400, 400

    
    
    #Sometimes, cap may not have initialized the capture. In that case, this code shows error. You can check whether it is initialized or not by the method cap.isOpened(). If it is True, OK. Otherwise open it using cap.open().
    while (cap.isOpened()):
        ret, img = cap.read()
        #flip the image horizontally
        img = cv2.flip(img, 1)
        
        gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray1,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40,40)
        )
                
        try: 
            x1,y1,w1,h1 = faces[0]
        except:
            continue
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(hsv, np.array([2, 50, 60]), np.array([25, 150, 255]))
        res = cv2.bitwise_and(img, img, mask=mask2)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        median = cv2.GaussianBlur(gray, (5, 5), 0)

        kernel_square = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(median, kernel_square, iterations=2)
        opening = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_square)
        ret, thresh = cv2.threshold(opening, 30, 255, cv2.THRESH_BINARY)

        thresh = thresh[y:y + h, x:x + w]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 2500:
                x, y, w1, h1 = cv2.boundingRect(contour)
                newImage = thresh[y:y + h1, x:x + w1]
                newImage = cv2.resize(newImage, (50, 50))
                pred_probab, pred_class = keras_predict(model, newImage)
                print(pred_class, pred_probab)
                if( pred_probab >= 0.96):
                    img = put_mask(images[pred_class], img, x1, y1, w1, h1)
               
        cv2.rectangle(img, (700,50), (1100, 450), (0, 255, 0), 2)
        x, y, w, h = 700, 50, 400, 400
        cv2.imshow("Mask", img)
        cv2.imshow("Shadows", thresh)
        k = cv2.waitKey(10)
        if k == 27:
            break

def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


def keras_process_image(img):
    image_x = 50
    image_y = 50
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img

def get_images():
    images_folder = 'shadow/'
    images = []
    for image in range(len(os.listdir(images_folder))):
        print(image)
        images.append(cv2.imread(images_folder+str(image)+'.png', -1))
    return images


keras_predict(model, np.zeros((50, 50, 1), dtype=np.uint8))
main()
