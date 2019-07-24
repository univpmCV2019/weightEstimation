import cv2
import numpy as np
import scipy.integrate as sInt
import os
import re

path = ''
outpath = ''

THRESH = 1/8
border = 10
minArea = 5550
minHeadArea = 2500
minShoulderArea = 4000
headTolerance = 0.07


def clear():
    if(os.name != 'nt'):
        os.system('clear')
    else:
        os.system('cls')


def findPerson(img):
    # Cropping and thresholding
    img = img[border:-border, border:-border]
    imgArea = img.shape[0]*img.shape[1]
    ret, thr = cv2.threshold(img, 7, 255, cv2.THRESH_TOZERO)
    temp, contours, hierarchy = cv2.findContours(thr, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

    # Finding contours of the person
    area_list = list()
    for i in contours:
        if cv2.contourArea(i) < imgArea*0.5 and cv2.contourArea(i) > minArea:
            area_list.append(i)
    area_list = sorted(area_list, reverse=False, key=lambda x: cv2.contourArea(x[0]))
    try:
        x, y, w, h = cv2.boundingRect(area_list[0])
        return x, y, w, h
    except IndexError:
        print("No person found")
        return None


def findBox(source):
    try:
        # Open an 8bit grayscale version
        gray = cv2.imread(source, cv2.COLORSPACE_GRAY)
        # Open the 16bit image, unchanged
        img = cv2.imread(source, cv2.IMREAD_UNCHANGED)
        imgArea = img.shape[0]*img.shape[1]
        # then convert it to 32bit float an normalize it
        img = np.array(img, dtype=np.float32)
        cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
    except Exception:
        print("error opening the image")
    try:
        # Launch the countour based segmentation
        x, y, w, h = findPerson(gray)
    except TypeError:
        # If None type returned, no person was found
        return

    # Cropping just to match the coordinates given by findPerson
    img = img[border:-border, border:-border]
    # Isolate the person. From now on, every operation is made only on this portion of the frame
    img = img[y:y+h, x:x+w]
    # Add a padding to ease further processing and avoid problems near the borders
    img = np.pad(img, 10, mode='constant', constant_values=img[0, 0])
    img = np.array(img, dtype=np.float32)
    cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
    basevol = sInt.simps(sInt.simps(np.ones(img.shape)))
    vol = sInt.simps(sInt.simps(img))/basevol
    globalMin, globalMax, globalMinP, globalMaxP = cv2.minMaxLoc(img)
    # Empirical threshold to isolate head and torso from the background
    THRESH = (globalMax-globalMin)*0.47
    ret, thr = cv2.threshold(img.copy(), THRESH, 1, cv2.THRESH_TOZERO_INV)
    temp, contours, hierarchy = cv2.findContours(np.uint8(thr.copy()*255), mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

    # save every value over 0
    backAndBorderMask = thr > 0
    # every value over 0 is set to 1
    backAndBorderMask[thr > 0] = 1
    # convert the mask to a 8 bit int array (to be shown and applied in the minMax operator)
    backAndBorderMask = np.array(backAndBorderMask, dtype=np.uint8)
    # global MIN is choosen to represent the top of the head
    globalMin, globalMax, globalMinP, globalMaxP = cv2.minMaxLoc(thr, backAndBorderMask)
    print("Minimo globale: "+str(globalMin)+", Massimo globale: "+str(globalMax))
    try:
        # Empirical mask to isolate head
        headMask = ((thr <= (globalMin+headTolerance)) & (thr >= (globalMin-headTolerance)))
        # isolate shoulders
        shouldersMask = (thr-headMask) > 0
        # better refine shoulders
        shMin, shMax, shMinP, shMaxP = cv2.minMaxLoc(thr, np.array(shouldersMask, dtype=np.uint8))
        shouldersMask = (thr <= (shMin))

        # cleaning sparse blobs
        shouldersMask = 255-cv2.dilate(cv2.medianBlur(np.array(shouldersMask*255, dtype=np.uint8), 13), (15, 15), 2)
        headMask = np.array(headMask*255, dtype=np.uint8)
    except Exception:
        return

    temp, headContour, hierarchyHead = cv2.findContours(headMask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    temp, shoulderContour, hierarchyShoulder = cv2.findContours(shouldersMask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    headFound = False
    for i in headContour:
        headArea = cv2.contourArea(i)
        if(len(i) >= 5 and headArea > minHeadArea):
            hEllipse = cv2.fitEllipse(i)
            cds = cv2.minAreaRect(i)
            cds = cv2.boxPoints(cds)
            # check aspect ratio to avoid considering distorted frames
            heaxis1 = np.sqrt((cds[0, 0]-cds[1, 0])**2+(cds[0, 1]-cds[1, 1])**2)
            heaxis2 = np.sqrt((cds[1, 0]-cds[2, 0])**2+(cds[1, 1]-cds[2, 1])**2)
            heAxes = np.sort([heaxis1, heaxis2])
            aspect_ratio = heAxes[1]/heAxes[0]
            if(aspect_ratio <= 1.4 and aspect_ratio >= 0.6):
                cv2.ellipse(thr, hEllipse, (255, 0, 0), 1)
                print("Volume: "+str(vol))
                print("Aspect ratio: "+str(aspect_ratio))
                print("Area testa: "+str(headArea))
                outFile.write(str(vol)+", "+str(headArea/imgArea)+", "+str(1/aspect_ratio)+", ")
                headFound = True
    if (not headFound):
        print("No head found")
        return
    try:
        # Concatenate all the contours found in the shoulder mask:
        # this is needed to "grab" both shoulders even when split apart by the head silhouette
        listCont = np.concatenate(shoulderContour[:])
        if (len(np.asarray(listCont)) >= 5):
            shEllipse = cv2.fitEllipse(listCont)
            cv2.ellipse(thr, shEllipse, (255, 0, 0), 2)
            shbox = cv2.boxPoints(shEllipse)
            shaxis1 = np.sqrt((shbox[0, 0]-shbox[1, 0])**2+(shbox[0, 1]-shbox[1, 1])**2)
            shaxis2 = np.sqrt((shbox[1, 0]-shbox[2, 0])**2+(shbox[1, 1]-shbox[2, 1])**2)
            shAxes = np.sort([shaxis1, shaxis2])
            # Shoulder area is noisy, we take an average of the height
            shHeight = cv2.mean(thr, np.array(shouldersMask*255, dtype=np.uint8))[0]
            print("Area spalle: "+str(cv2.contourArea(listCont))+", assi: "+str(shAxes))
            outFile.write(str(cv2.contourArea(listCont)/imgArea)+", "+str(shAxes[0]/gray.shape[0])+", "+str(shAxes[1]/gray.shape[0])+", "+str(shHeight)+", ")
        else:
            print("")
            outFile.write("error, ")
    except ValueError:
        print("No shoulders found")
        # Printing 'error' on output file to inform that this line is unreliable
        outFile.write("error, ")
        pass

    if(headFound):
        # Height as average of head mask value
        height = cv2.mean(thr, np.array(headMask*255, dtype=np.uint8))[0]
        print("Altezza: "+str(height))
        outFile.write(str(height)+", ")
        outFile.write(re.sub('_.*', '', str(os.path.basename(source)))+"\n")
    # DEBUG
    # cv2.imshow("thr", thr)
    # cv2.waitKey(3)


outFile = open(outpath+"out_measures.txt", 'w')
outFile.write("volume, head area, aspect ratio, shoulder area, axis1, axis2, shoulder height, height, index\n\n")

last = ""
for root, dirs, files in os.walk(path):
    for i in sorted(files):
        if '16bit' in i:
            findBox(path+i)
            if last != re.sub('_.*', '', str(os.path.basename(path+i))):
                outFile.write("\n")
                last = re.sub('_.*', '', str(os.path.basename(path+i)))
outFile.close()
