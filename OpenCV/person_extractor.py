import cv2
import numpy as np
import os

path = ''
outpath = ''
border = 10
minArea = 5050
headRadius = 5


def clear():
    if(os.name != 'nt'):
        os.system('clear')
    else:
        os.system('cls')


def findBox(source):
    try:
        img = cv2.imread(source, cv2.COLORSPACE_GRAY)
    except Exception:
        print("Error opening the image")
    # Cropping and thresholding
    img = img[border:-border, border:-border]
    ret, thr = cv2.threshold(img, 7, 256, cv2.THRESH_TOZERO)
    # Contours
    img2, contours, hierarchy = cv2.findContours(thr, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    # Ruling out the biggest contour, as it matches the image borders
    area_list = list()
    for i in contours:
        area_list.append(cv2.contourArea(i))
    maxArea = max(area_list)
    for i in contours:
        area = cv2.contourArea(i)
        # if area is not as big as the image borders, but big enough
        if (area >= minArea) & (area < maxArea):
            # find the bounding box surrounding the contour
            x, y, w, h = cv2.boundingRect(i)
            # blank=np.zeros((int(img.shape[0]/2),int(img.shape[1]/2)),dtype=img.dtype)
            # ROI/blank too small, trying 300x404 (biggest ROI)
            blank = np.zeros((300, 404), dtype=img.dtype)
            # copy the ROI in a blank frame
            blank[0:h, 0:w] = img[y:y+h, x:x+w]
            # write the output image
            cv2.imwrite(outpath+str(os.path.basename(source)), blank)


for root, dirs, files in os.walk(path):
    for i in sorted(files):
        # print(i)
        if '16bit' in i:
            findBox(path+i)
