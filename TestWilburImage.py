# -*- coding: utf-8 -*-
"""
Created on May 2023

@author: Alfonso Blanco
"""

import time
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

import cv2

X_resize=220
Y_resize=70

import imutils


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

###########################################################
# MAIN
##########################################################
Ini=time.time()
gray = cv2.imread("WilburImage.jpg")
#cv2.imshow("gray",gray)
#cv2.waitKey()
grayColor=gray

gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
   
TotHits=0  

X_resize=215
Y_resize=70
 

gray=cv2.resize(gray,None,fx=1.78,fy=1.78,interpolation=cv2.INTER_CUBIC)

gray = cv2.resize(gray, (X_resize,Y_resize), interpolation = cv2.INTER_AREA)

rotation, spectrum, frquency =GetRotationImage(gray)
rotation=90.0 - rotation 
#rotation=10.09
if (rotation > 0 and rotation < 30)  or (rotation < 0 and rotation > -30):
    print(" rotate "+ str(rotation))
    gray=imutils.rotate(gray,angle=rotation)


kernel = np.ones((2,2),np.uint8)

gray = cv2.GaussianBlur(gray, (3, 3), 0)

gray = cv2.dilate(gray,kernel,iterations = 1)

   
# https://medium.com/practical-data-science-and-engineering/image-kernels-88162cb6585d
   
kernel = np.array([[0, -1, 0],
              [-1,10, -1],
              [0, -1, 0]])
dst = cv2.filter2D(gray, -1, kernel)
img_concat = cv2.hconcat([gray, dst])

text, Accuraccy = GetPaddleOcr(img_concat)
text = ''.join(char for char in text if char.isalnum())
text=ProcessText(text)

print(" License detected = "+ text)
print(" Time in seconds " + str(time.time() - Ini))