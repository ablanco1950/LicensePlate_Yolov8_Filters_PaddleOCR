# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 20:1 7:29 2022

@author: Alfonso Blanco
"""
#######################################################################
# PARAMETERS
######################################################################
dir=""

#dirname= "test6Training\\images"
#dirname= "archiveLabeled"
#dirname= "C:\\Malos\\images"
dirname= "Test"
#dirname= "roboflow\\test\\images"
#https://github.com/mrzaizai2k/VIETNAMESE_LICENSE_PLATE/tree/master/data/image
#dirname="C:\\PruebasGithub\\License-Plate-Recognition-YOLOv7-and-CNN-main\\License-Plate-Recognition-YOLOv7-and-CNN-main\\data\\test\\images"
#dirname="C:\\PruebasGithub\\LicensePlateDetector-master\\LicensePlateDetector-master\\output"
#dirname="C:\\PruebasGithub\\detectron2-licenseplates-master\\detectron2-licenseplates-master\\datasets\\licenseplates\\images"


import cv2

# suggested by Wilbur
# https://github.com/Saafke/FSRCNN_Tensorflow/tree/master/models
# https://learnopencv.com/super-resolution-in-opencv/#sec5
# https://learnopencv.com/super-resolution-in-opencv/
ocv_model = cv2.dnn_superres.DnnSuperResImpl_create()
ocv_weight = 'FSRCNN_x4.pb'
ocv_model.readModel(ocv_weight)
ocv_model.setModel('fsrcnn', 4)


import time
Ini=time.time()


dirnameYolo="runs\\detect\\train9\\weights\\best.pt"
# https://docs.ultralytics.com/python/
from ultralytics import YOLO
model = YOLO(dirnameYolo)
class_list = model.model.names
#print(class_list)


######################################################################
from paddleocr import PaddleOCR
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
# to switch the language model in order.
# https://pypi.org/project/paddleocr/
#
# supress anoysing logging messages parameter show_log = False
# https://github.com/PaddlePaddle/PaddleOCR/issues/2348
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log = False) # need to run only once to download and load model into memory

import numpy as np

X_resize=220
Y_resize=70

import os
import re

import imutils

from scipy.signal import convolve2d

# Control filters accuracy
TabTotHitsFilter=[]
TabTotFailuresFilter=[]

for j in range(60):
    TabTotHitsFilter.append(0)
    TabTotFailuresFilter.append(0)
#####################################################################
"""
Copied from https://gist.github.com/endolith/334196bac1cac45a4893#

other source:
    https://stackoverflow.com/questions/46084476/radon-transformation-in-python
"""

from skimage.transform import radon

import numpy
from numpy import  mean, array, blackman, sqrt, square
from numpy.fft import rfft


# https://gist.github.com/fubel/ad01878c5a08a57be9b8b80605ad1247
# comment from  nafe93 modified by Alfonso Blanco
def Otra_discrete_radon_transform(img): 
   #steps= img.shape[1] 
   steps= 90
   import skimage     
   # shape
   h, w = img.shape
   zero = np.zeros((h, steps), dtype='float64')
   # sum and roatate
   for s in range(steps):
   #for s in range(300):
      #if s > img.shape[0] -1: break 
      #rotation = skimage.transform.rotate(img, s, reshape=False).astype('float64')
      rotation = skimage.transform.rotate(img, s).astype('float64')
      # sum 
      #zero[:, s] = np.sum(rotation, axis=0)
      zero[:, s] = np.sum(rotation)
      # rotate image 
      #zero = skimage.transform.rotate(zero, 180).astype('float64') 
   print(zero.shape)
   return zero



try:
    # More accurate peak finding from
    # https://gist.github.com/endolith/255291#file-parabolic-py
    from parabolic import parabolic

    def argmax(x):
        return parabolic(x, numpy.argmax(x))[0]
except ImportError:
    from numpy import argmax


def GetRotationImage(image):

   
    I=image
    I = I - mean(I)  # Demean; make the brightness extend above and below zero
    
    
    # Do the radon transform and display the result
    sinogram = radon(I)
    # a   radon from scratch less efficient than radom
    #sinogram=Otra_discrete_radon_transform(I)
    
    
    # Find the RMS value of each row and find "busiest" rotation,
    # where the transform is lined up perfectly with the alternating dark
    # text and white lines
      
    # rms_flat does no exist in recent versions
    #r = array([mlab.rms_flat(line) for line in sinogram.transpose()])
    r = array([sqrt(mean(square(line))) for line in sinogram.transpose()])
    rotation = argmax(r)
    #print('Rotation: {:.2f} degrees'.format(90 - rotation))
    #plt.axhline(rotation, color='r')
    
    # Plot the busy row
    row = sinogram[:, rotation]
    N = len(row)
    
    # Take spectrum of busy row and find line spacing
    window = blackman(N)
    spectrum = rfft(row * window)
    
    frequency = argmax(abs(spectrum))
   
    return rotation, spectrum, frequency

 
 
# Copied from https://learnopencv.com/otsu-thresholding-with-opencv/ 
def OTSU_Threshold(image):
# Set total number of bins in the histogram

    bins_num = 256
    
    # Get the image histogram
    
    hist, bin_edges = np.histogram(image, bins=bins_num)
   
    # Get normalized histogram if it is required
    
    #if is_normalized:
    
    hist = np.divide(hist.ravel(), hist.max())
    
     
    
    # Calculate centers of bins
    
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
    
    
    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    
    weight1 = np.cumsum(hist)
    
    weight2 = np.cumsum(hist[::-1])[::-1]
   
    # Get the class means mu0(t)
    
    mean1 = np.cumsum(hist * bin_mids) / weight1
    
    # Get the class means mu1(t)
    
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
    
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    
    # Maximize the inter_class_variance function val
    
    index_of_max_val = np.argmax(inter_class_variance)
    
    threshold = bin_mids[:-1][index_of_max_val]
    
    #print("Otsu's algorithm implementation thresholding result: ", threshold)
    return threshold


# https://stackoverflow.com/questions/48213278/implementing-otsu-binarization-from-scratch-python
# answer 14
def Otsu2Values(gray):
    import math
    Media=np.mean(gray)
    Desv=np.std(gray)
    OcurrenciasMax=0
    ValorMax=0
    OcurrenciasMin=0
    ValorMin=0
    his, bins = np.histogram(gray, np.array(range(0, 256)))
    for i in range(len(bins)-1):
        #print ( " Ocurrencias " + str(his[i]) + " Valor " + str(bins[i]))
        if bins[i] > Media:
            if his[i] > OcurrenciasMax:
                OcurrenciasMax=his[i]
                ValorMax=bins[i]
        else:
            if his[i] > OcurrenciasMin:
                OcurrenciasMin=his[i]
                ValorMin=bins[i]
    
    #return ValorMax+(math.sqrt(Desv)/2), ValorMin-(math.sqrt(Desv)/2)
    final_img = gray.copy()
    #print(final_thresh)
    final_img[gray > ( ValorMax-Desv)] = 255
    final_img[gray < (ValorMin+Desv)] = 0
    return final_img
#########################################################################
def ApplyCLAHE(gray):
#https://towardsdatascience.com/image-enhancement-techniques-using-opencv-and-python-9191d5c30d45
    
    gray_img_eqhist=cv2.equalizeHist(gray)
    hist=cv2.calcHist(gray_img_eqhist,[0],None,[256],[0,256])
    clahe=cv2.createCLAHE(clipLimit=200,tileGridSize=(3,3))
    gray_img_clahe=clahe.apply(gray_img_eqhist)
    return gray_img_clahe


def GetPaddleOcr(img):

    """
    Created on Tue Mar  7 10:31:09 2023
    
    @author: https://pypi.org/project/paddleocr/ (adapted from)
    """
        
    cv2.imwrite("gray.jpg",img)
    img_path = 'gray.jpg'
    #cv2.imshow("gray",img)
    #cv2.waitKey()
    result = ocr.ocr(img_path, cls=True)
    for idx in range(len(result)):
        res = result[idx]
        #for line in res:
        #    print(line)
    
    # draw result
    from PIL import Image
    licensePlate= ""
    accuracy=0.0
    for i in range(len(result)):
        result = result[i]
        #image = Image.open(img_path).convert('RGB')
        boxes = [line[0] for line in result]
        
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        
        
        #print("RESULTADO  "+ str(txts))
        #print("confiabilidad  "+ str(scores))
        if len(txts) > 0:
            for j in range( len(txts)):
               licensePlate= licensePlate + txts[j]
            accuracy=float(scores[0])
    #print("SALIDA " + licensePlate)
    #print(accuracy)
      
    return licensePlate, accuracy

# https://medium.com/@garciafelipe03/image-filters-and-morphological-operations-using-python-89c5bbb8dca0
# 5x5 Gaussian Blur
def gaussian_5x5(img):
    
    kernel_gb_5 = (1 / 273) * np.array([[1, 4, 7, 4, 1],
                                        [4, 16, 26, 16, 4],
                                        [7, 26, 41, 26, 7],
                                        [4, 16, 26, 16, 4],
                                        [1, 4, 7, 4, 1]])

    return convolve2d(img, kernel_gb_5, 'valid')



#########################################################################
def FindLicenseNumber (gray, x_offset, y_offset,  License, x_resize, y_resize, \
                       Resize_xfactor, Resize_yfactor, BilateralOption):
#########################################################################

    grayColor=gray
    
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
   
    TotHits=0 
    
    X_resize=x_resize
    Y_resize=y_resize
     
    
    gray=cv2.resize(gray,None,fx=Resize_xfactor,fy=Resize_yfactor,interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.resize(gray, (X_resize,Y_resize), interpolation = cv2.INTER_AREA)
    
    
    # en la mayoria de los casos no es necesaria rotacion
    # pero en algunos casos si (ver TestRadonWithWilburImage.py)
    rotation, spectrum, frquency =GetRotationImage(gray)
    rotation=90 - rotation
    
    if (rotation > 0 and rotation < 30)  or (rotation < 0 and rotation > -30):
        print(License + " rotate "+ str(rotation))
        gray=imutils.rotate(gray,angle=rotation)
    
    
    TabLicensesFounded=[]
    ContLicensesFounded=[]
    
    
    X_resize=x_resize
    Y_resize=y_resize
    print("gray.shape " + str(gray.shape)) 
    Resize_xfactor=1.5
    Resize_yfactor=1.5
    
   
    TabLicensesFounded=[]
    ContLicensesFounded=[]
    
    TotHits=0
    
    
    gray1=gaussian_5x5(gray)
    text, Accuraccy = GetPaddleOcr(gray1)
    text = ''.join(char for char in text if char.isalnum())
    text=ProcessText(text)
    if ProcessText(text) != "":
    
           TabLicensesFounded, ContLicensesFounded =ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
           if text==License:
              print(text + "  Hit with gaussian_5x5 " )
              TotHits=TotHits+1
           else:
               print(License + " detected with Filter gaussian_5x5 "+ text) 
    
    
        
    # https://medium.com/practical-data-science-and-engineering/image-kernels-88162cb6585d
    kernel = np.ones((2,2),np.uint8)
    
    gray1 = cv2.GaussianBlur(gray, (3, 3), 0)
    gray1 = cv2.dilate(gray1,kernel,iterations = 1)
    for z in range (10,11):
        
       
        kernel = np.array([[0, -1, 0],
                       [-1,z, -1],
                       [0, -1, 0]])
        dst = cv2.filter2D(gray1, -1, kernel)
        img_concat = cv2.hconcat([gray1, dst])
        text, Accuraccy = GetPaddleOcr(img_concat)
        text = ''.join(char for char in text if char.isalnum())
        text=ProcessText(text)
        if ProcessText(text) != "":
        
               TabLicensesFounded, ContLicensesFounded =ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
               if text==License:
                  print(text + "  Hit with image concat  z= " + str(z))
                  TotHits=TotHits+1
               else:
                   print(License + " detected with Filter image concat "+ text+ " z= "+ str(z)) 
        
       
       
    
    gray1= ocv_model.upsample(gray)
    #cv2.imshow("Ocv",gray1)
    #cv2.waitKey()
    text, Accuraccy = GetPaddleOcr(gray1)    
    text = ''.join(char for char in text if char.isalnum())
    text=ProcessText(text)
    if ProcessText(text) != "":
     
            TabLicensesFounded, ContLicensesFounded =ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
            if text==License:
               print(text + "  Hit with FSRCNN  ")
               TotHits=TotHits+1
               TabTotHitsFilter[0]=TabTotHitsFilter[0]+1
            else:
                print(License + " detected with Filter FSRCNN "+ text) 
                TabTotFailuresFilter[0]=TabTotFailuresFilter[0]+1
    
    
  
    
   
    
    gray1=Otsu2Values(gray)
    #cv2.imshow("Otsu2",gray1)
    #cv2.waitKey()
    text, Accuraccy = GetPaddleOcr(gray1)    
    text = ''.join(char for char in text if char.isalnum())
    text=ProcessText(text)
    if ProcessText(text) != "":
     
            TabLicensesFounded, ContLicensesFounded =ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
            if text==License:
               print(text + "  Hit with Otsu2Values  ")
               TotHits=TotHits+1
               #TabTotHitsFilter[0]=TabTotHitsFilter[0]+1
            else:
                print(License + " detected with Filter Otsu2Values "+ text) 
                #TabTotFailuresFilter[0]=TabTotFailuresFilter[0]+1
    
    
    # threshold contrast, author Alfonso Blanco
    # usefull in some cases
    #print ("Media " + str(int(np.mean(gray))) + " Desviacion "+str(np.std(gray)))
    ret, thresh = cv2.threshold(gray, int(np.mean(gray)-  np.std(gray) ), int(np.mean(gray)+  np.std(gray) ), cv2.THRESH_BINARY)
    
    #cv2.imshow("thr contrast",thresh)
    #cv2.waitKey()
    text, Accuraccy = GetPaddleOcr(thresh)
    text = ''.join(char for char in text if char.isalnum())
    text=ProcessText(text)
    if ProcessText(text) != "":
     
            TabLicensesFounded, ContLicensesFounded =ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
            if text==License:
               print(text + "  Hit with threshold contrast  ")
               TotHits=TotHits+1
               #TabTotHitsFilter[0]=TabTotHitsFilter[0]+1
            else:
                print(License + " detected with Filter threshold contrast "+ text) 
                #TabTotFailuresFilter[0]=TabTotFailuresFilter[0]+1
    
    ################################################
    # https://medium.com/@sahilutekar.su/unlocking-the-full-potential-of-images-with-python-and-opencv-a-step-by-step-guide-to-advanced-7f0a8618c732
    # equalization
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_output = clahe.apply(equalized)
    #cv2.imshow('Equalized', equalized)
    #cv2.imshow('CLAHE Output', clahe_output)

    # Calculate the high-pass filtered image
    high_pass = cv2.subtract(gray, blur)

    # Add the high-pass filtered image back to the original image
    sharpened = cv2.addWeighted(gray, 1.2, high_pass, -1.5, 0)
    #cv2.imshow('equalization', sharpened)
    #cv2.waitKey()
    text, Accuraccy = GetPaddleOcr(sharpened)
    text = ''.join(char for char in text if char.isalnum())
    text=ProcessText(text)
    if ProcessText(text) != "":
    
           TabLicensesFounded, ContLicensesFounded =ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
           if text==License:
              print(text + "  Hit with equalization  ")
              TotHits=TotHits+1
           else:
               print(License + " detected with Filter equalization "+ text) 
    
   
    
    
    kernel = np.ones((3,3),np.float32)/90
    gray1 = cv2.filter2D(gray,-1,kernel)   
    #gray_clahe = cv2.GaussianBlur(gray, (5, 5), 0) 
    gray_img_clahe=ApplyCLAHE(gray1)
    
    th=OTSU_Threshold(gray_img_clahe)
    max_val=255
    
    ret, o3 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TOZERO)
    text, Accuraccy = GetPaddleOcr(o3)
    
    text = ''.join(char for char in text if char.isalnum()) 
    text=ProcessText(text)
    if ProcessText(text) != "":
    
            TabLicensesFounded, ContLicensesFounded =ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
            if text==License:
               print(text + "  Hit with CLAHE  and THRESH_TOZERO" )
               #TotHits=TotHits+1
            else:
                print(License + " detected with CLAHE and THRESH_TOZERO as "+ text) 
    
    
    
    for z in range(5,6):
   
    
       kernel = np.array([[0,-1,0], [-1,z,-1], [0,-1,0]])
       gray1 = cv2.filter2D(gray, -1, kernel)
              
       text, Accuraccy = GetPaddleOcr(gray1)
       
       text = ''.join(char for char in text if char.isalnum()) 
       text=ProcessText(text)
       if ProcessText(text) != "":
      
           ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
           if text==License:
              print(text +  "  Hit with Sharpen filter z= "  +str(z))
              TotHits=TotHits+1
           else:
               print(License + " detected with Sharpen filter z= "  +str(z) + " as "+ text) 
      
    
    gray_img_clahe=ApplyCLAHE(gray)
    
    th=OTSU_Threshold(gray_img_clahe)
    max_val=255
    
    #   Otsu's thresholding
    ret2,gray1 = cv2.threshold(gray,0,255,cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
    #gray1 = cv2.GaussianBlur(gray1, (1, 1), 0)
    text, Accuraccy = GetPaddleOcr(gray1)
          
    text = ''.join(char for char in text if char.isalnum())
    
    text=ProcessText(text)
    if ProcessText(text) != "":
            TabLicensesFounded, ContLicensesFounded =ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
            if text==License:
               print(text + "  Hit with Otsu's thresholding of cv2 and THRESH_TRUNC" )
               TotHits=TotHits+1
            else:
                print(License + " detected with Otsu's thresholding of cv2 and THRESH_TRUNC as "+ text) 
   
    
    ####################################################
    # experimental formula based on the brightness
    # of the whole image 
    ####################################################
    
    SumBrightness=np.sum(gray)  
    threshold=(SumBrightness/177600.00) 
    
    #####################################################
     
    ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TOZERO)
    #gray1 = cv2.GaussianBlur(gray1, (1, 1), 0)
    text, Accuraccy = GetPaddleOcr(gray1)
   
    text = ''.join(char for char in text if char.isalnum())
    
    text=ProcessText(text)
    if ProcessText(text) != "":
        ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
        if text==License:
           print(text + "  Hit with Brightness and THRESH_TOZERO" )
           TotHits=TotHits+1
        else:
            print(License + " detected with Brightness and THRESH_TOZERO as "+ text)
    
   
    ################################################################
    return TabLicensesFounded, ContLicensesFounded

 ########################################################################
def loadimagesRoboflow (dirname):
 #########################################################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco García
 ########################################################################  
     imgpath = dirname + "\\"
     
     images = []
     Licenses=[]
    
     print("Reading imagenes from ",imgpath)
     NumImage=-2
     
     Cont=0
     for root, dirnames, filenames in os.walk(imgpath):
        
         NumImage=NumImage+1
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                 License=filename[:len(filename)-4]
                 #if License != "PGMN112": continue
                 
                 image = cv2.imread(filepath)
                 # Roboflow images are (416,416)
                 #image=cv2.resize(image,(416,416)) 
                 # kaggle images
                 #image=cv2.resize(image, (640,640))
                 
                           
                 images.append(image)
                 Licenses.append(License)
                 
                 Cont+=1
     
     return images, Licenses

def Detect_International_LicensePlate(Text):
    if len(Text) < 3 : return -1
    for i in range(len(Text)):
        if (Text[i] >= "0" and Text[i] <= "9" )   or (Text[i] >= "A" and Text[i] <= "Z" ):
            continue
        else: 
          return -1 
       
    return 1

def ProcessText(text):
  
    if len(text)  > 10:
        text=text[len(text)-10]
        if len(text)  > 9:
          text=text[len(text)-9]
        else:
            if len(text)  > 8:
              text=text[len(text)-8]
            else:
        
                if len(text)  > 7:
                   text=text[len(text)-7:] 
    if Detect_International_LicensePlate(text)== -1: 
       return ""
    else:
       return text

def ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text):
    
    SwFounded=0
    for i in range( len(TabLicensesFounded)):
        if text==TabLicensesFounded[i]:
            ContLicensesFounded[i]=ContLicensesFounded[i]+1
            SwFounded=1
            break
    if SwFounded==0:
       TabLicensesFounded.append(text) 
       ContLicensesFounded.append(1)
    return TabLicensesFounded, ContLicensesFounded


# ttps://medium.chom/@chanon.krittapholchai/build-object-detection-gui-with-yolov8-and-pysimplegui-76d5f5464d6c
def DetectLicenseWithYolov8 (img):
  
   TabcropLicense=[]
   y=[]
   yMax=[]
   x=[]
   xMax=[]
   results = model.predict(img)
   for i in range(len(results)):
       # may be several plates in a frame
       result=results[i]
       
       xyxy= result.boxes.xyxy.numpy()
       confidence= result.boxes.conf.numpy()
       class_id= result.boxes.cls.numpy().astype(int)
       # Get Class name
       class_name = [class_list[z] for z in class_id]
       # Pack together for easy use
       sum_output = list(zip(class_name, confidence,xyxy))
       # Copy image, in case that we need original image for something
       out_image = img.copy()
       for run_output in sum_output :
           # Unpack
           #print(class_name)
           label, con, box = run_output
           if label == "vehicle":continue
           cropLicense=out_image[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
           #cv2.imshow("Crop", cropLicense)
           #cv2.waitKey(0)
           TabcropLicense.append(cropLicense)
           y.append(int(box[1]))
           yMax.append(int(box[3]))
           x.append(int(box[0]))
           xMax.append(int(box[2]))
       
   return TabcropLicense, y,yMax,x,xMax


###########################################################
# MAIN
##########################################################

imagesComplete, Licenses=loadimagesRoboflow(dirname)

print("Number of imagenes : " + str(len(imagesComplete)))

print("Number of   licenses : " + str(len(Licenses)))

ContDetected=0
ContNoDetected=0
TotHits=0
TotFailures=0
with open( "LicenseResults.txt" ,"w") as  w:
    for i in range (len(imagesComplete)):
          
            gray=imagesComplete[i]
            
            License=Licenses[i]
            #gray1, gray = Preprocess.preprocess(gray)
            TabImgSelect, y, yMax, x, xMax =DetectLicenseWithYolov8(gray)
            
            if TabImgSelect==[]:
                print(License + " NON DETECTED")
                ContNoDetected=ContNoDetected+1 
                continue
            else:
                ContDetected=ContDetected+1
                print(License + " DETECTED ")
            for x in range(len(TabImgSelect)):
                if TabImgSelect[x] == []: continue
                gray=TabImgSelect[x]  
                #if len(TabImgSelect) > 1:
                #    gray=TabImgSelect[1]
                #cv2.imshow('Frame', gray)
                
                x_off=3
                y_off=2
                
                #x_resize=220
                x_resize=215
                y_resize=70
                
                Resize_xfactor=1.78
                Resize_yfactor=1.78
                
                ContLoop=0
                
                SwFounded=0
                
                BilateralOption=0
                TabLicensesFounded=[]
                ContLicensesFounded=[]
                
                TabLicensesFounded, ContLicensesFounded= FindLicenseNumber (gray, x_off, y_off,  License, x_resize, y_resize, \
                                       Resize_xfactor, Resize_yfactor, BilateralOption)
                  
                
                print(TabLicensesFounded)
                print(ContLicensesFounded)
                
                ymax=-1
                contmax=0
                licensemax=""
              
                for z in range(len(TabLicensesFounded)):
                    if ContLicensesFounded[z] > contmax:
                        contmax=ContLicensesFounded[z]
                        licensemax=TabLicensesFounded[z]
                
                if licensemax == License:
                   print(License + " correctly recognized") 
                   TotHits+=1
                else:
                    print(License + " Detected but not correctly recognized")
                    TotFailures +=1
                print ("")  
                lineaw=[]
                lineaw.append(License) 
                lineaw.append(licensemax)
                lineaWrite =','.join(lineaw)
                lineaWrite=lineaWrite + "\n"
                w.write(lineaWrite)
             
              
print("")           
print("Total Hits = " + str(TotHits ) + " from " + str(len(imagesComplete)) + " images readed")

print("")

#print("Total Hits filtro nuevo= " + str(TabTotHitsFilter[0] ) + " from " + str(len(imagesComplete)) + " images readed")
#print("Total Failures filtro nuevo= " + str(TabTotFailuresFilter[0] ) + " from " + str(len(imagesComplete)) + " images readed")

print( " Time in seconds "+ str(time.time()-Ini))
