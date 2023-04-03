# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 20:1 7:29 2022

@author: Alfonso Blanco
"""
######################################################################
# PARAMETERS
#####################################################################
#####################################################################
API_KEY=""

######################################################
# Video from https://github.com/anmspro/Traffic-Signal-Violation-Detection-System/tree/master/Resources
dirVideo="Traffic IP Camera video.mp4"
#dirVideo="vid.mp4"
# in  14 minutes = 800 seconds finish  
TimeLimit=1000
# Max number of Snapshots to consider a image
# lower 5 snapshots is consider noisy
LimitSnapshot=5
# to increase the speed of the process,
# even if some license plates are lost,
# only one snapshot out of every SpeedUpSnapshot is processed
SpeedUpSnapshot=10
##############################################################
# DOWNLOAD VIDEOS TO TEST
###############################################################
# video from https://github.com/hasaan21/Car-Number-Plate-Recognition-Sysytem
#dirVideo="vid.mp4"

#dirVideo="video12.mp4"
#dirVideo="C:\\Car_Speed_Detection\\Comma.ai.Data.and.Model\\Comma.ai Model\\train.mp4"

# from https://www.pexels.com/video/video-of-famous-landmark-on-a-city-during-daytime-1721294/
#dirVideo="Pexels Videos 1721294.mp4"

import json

"""
@author: https://blog.roboflow.com/how-to-crop-computer-vision-model-predictions/
"""

from roboflow import Roboflow

rf = Roboflow(api_key=API_KEY)
project = rf.workspace().project("license-plate-recognition-rxg4e")
model = project.version(4).model



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

import time

TimeIni=time.time()


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

#####################################################################
def ThresholdStable(image):
    # -*- coding: utf-8 -*-
    """
    Created on Fri Aug 12 21:04:48 2022
    Author: Alfonso Blanco García
    
    Looks for the threshold whose variations keep the image STABLE
    (there are only small variations with the image of the previous 
     threshold).
    Similar to the method followed in cv2.MSER
    https://datasmarts.net/es/como-usar-el-detector-de-puntos-clave-mser-en-opencv/https://felipemeganha.medium.com/detecting-handwriting-regions-with-opencv-and-python-ff0b1050aa4e
    """
  
    thresholds=[]
    Repes=[]
    Difes=[]
    
    gray=image 
    grayAnt=gray

    ContRepe=0
    threshold=0
    for i in range (255):
        
        ret, gray1=cv2.threshold(gray,i,255,  cv2.THRESH_BINARY)
        Dife1 = grayAnt - gray1
        Dife2=np.sum(Dife1)
        if Dife2 < 0: Dife2=Dife2*-1
        Difes.append(Dife2)
        if Dife2<22000: # Case only image of license plate
        #if Dife2<60000:    
            ContRepe=ContRepe+1
            
            threshold=i
            grayAnt=gray1
            continue
        if ContRepe > 0:
            
            thresholds.append(threshold) 
            Repes.append(ContRepe)  
        ContRepe=0
        grayAnt=gray1
    thresholdMax=0
    RepesMax=0    
    for i in range(len(thresholds)):
        #print ("Threshold = " + str(thresholds[i])+ " Repeticiones = " +str(Repes[i]))
        if Repes[i] > RepesMax:
            RepesMax=Repes[i]
            thresholdMax=thresholds[i]
            
    #print(min(Difes))
    #print ("Threshold Resultado= " + str(thresholdMax)+ " Repeticiones = " +str(RepesMax))
    return thresholdMax

 
 
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
    
    
    result = ocr.ocr(img_path,  cls=True)
    # draw result
    #from PIL import Image
    result = result[0]
    #image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    
    licensePlate= ""
    accuracy=0.0
    #print("RESULTADO  "+ str(txts))
    #print("confiabilidad  "+ str(scores))
    if len(txts) > 0:
        licensePlate= txts[0]
        accuracy=float(scores[0])
    #print(licensePlate)
    #print(accuracy)
      
    return licensePlate, accuracy
    
   
#########################################################################
def FindLicenseNumber (gray, x_offset, y_offset,  License, x_resize, y_resize, \
                       Resize_xfactor, Resize_yfactor, BilateralOption):
#########################################################################

    
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
   
    TotHits=0  
    
    X_resize=x_resize
    Y_resize=y_resize
     
    
    gray=cv2.resize(gray,None,fx=Resize_xfactor,fy=Resize_yfactor,interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.resize(gray, (X_resize,Y_resize), interpolation = cv2.INTER_AREA)
    
    rotation, spectrum, frquency =GetRotationImage(gray)
    rotation=90 - rotation
    #print("Car" + str(NumberImageOrder) + " Brillo : " +str(SumBrightnessLic) +   
    #      " Desviacion : " + str(DesvLic))
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
    AccuraccyMin=0.7
    TotHits=0
  
    # https://medium.com/practical-data-science-and-engineering/image-kernels-88162cb6585d
    #kernel = np.array([[0, -1, 0],
    #               [-1, 5, -1],
    #               [0, -1, 0]])
   
    
    kernel = np.array([[0, -1, 0],
                   [-1,10, -1],
                   [0, -1, 0]])
    dst = cv2.filter2D(gray, -1, kernel)
    img_concat = cv2.hconcat([gray, dst])
    text, Accuraccy = GetPaddleOcr(img_concat)
    if Accuraccy < AccuraccyMin:
        text=""
    text = ''.join(char for char in text if char.isalnum())
    text=ProcessText(text)
    if ProcessText(text) != "":
    
           TabLicensesFounded, ContLicensesFounded =ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
           if text==License:
              print(text + "  Hit with image concat  ")
              TotHits=TotHits+1
           else:
               print(License + " detected with Filter image concat "+ text) 
    
    
    
    kernel = np.ones((3,3),np.float32)/90
    gray1 = cv2.filter2D(gray,-1,kernel)   
    #gray_clahe = cv2.GaussianBlur(gray, (5, 5), 0) 
    gray_img_clahe=ApplyCLAHE(gray1)
    
    th=OTSU_Threshold(gray_img_clahe)
    max_val=255
    
    ret, o3 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TOZERO)
    text, Accuraccy = GetPaddleOcr(o3)
    if Accuraccy < AccuraccyMin:
        text=""
    
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
       if Accuraccy < AccuraccyMin:
           text=""
       
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
    if Accuraccy < AccuraccyMin:
        text=""
          
    text = ''.join(char for char in text if char.isalnum())
    text=ProcessText(text)
    if ProcessText(text) != "":
    #if Detect_International_LicensePlate(text)== 1:
            TabLicensesFounded, ContLicensesFounded =ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
            if text==License:
               print(text + "  Hit with Otsu's thresholding of cv2 and THRESH_TRUNC" )
               TotHits=TotHits+1
            else:
                print(License + " detected with Otsu's thresholding of cv2 and THRESH_TRUNC as "+ text) 
   
    
    threshold=ThresholdStable(gray)
    ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TRUNC) 
    #gray1 = cv2.GaussianBlur(gray1, (1, 1), 0)
    text, Accuraccy = GetPaddleOcr(gray1)
    if Accuraccy < AccuraccyMin:
        text=""
   
    text = ''.join(char for char in text if char.isalnum())
    text=ProcessText(text)
    if ProcessText(text) != "":
    #if Detect_International_LicensePlate(text)== 1:
        ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)    
        if text==License:
            print(text + "  Hit with Stable and THRESH_TRUNC" )
            TotHits=TotHits+1
        else:
            print(License + " detected with Stable and THRESH_TRUNC as "+ text)         
     
    
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
    if Accuraccy < AccuraccyMin:
        text=""
   
    text = ''.join(char for char in text if char.isalnum())
    text=ProcessText(text)
    if ProcessText(text) != "":
    #if Detect_International_LicensePlate(text)== 1:
        ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
        if text==License:
           print(text + "  Hit with Brightness and THRESH_TOZERO" )
           TotHits=TotHits+1
        else:
            print(License + " detected with Brightness and THRESH_TOZERO as "+ text)
    
    
    ################################################################
    return TabLicensesFounded, ContLicensesFounded


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


def DetectLicenseWithRoboflow (img):
  
    TabcropLicense=[]
    #img = cv2.resize(img, (640, 640), interpolation = cv2.INTER_AREA)
    cv2.imwrite("gray.jpg",img)
    img_path = 'gray.jpg'
    inference = model.predict(img_path, confidence=40)
    #print(inference[0])
    #json_load=inference
    if str(inference)=="":
        return TabcropLicense,0,0,0,0
        
    # json.loads only admit one inference for that inference[0]
    # https://stackoverflow.com/questions/21058935/python-json-loads-shows-valueerror-extra-data
    # answer 14.1
    json_load = (json.loads(str(inference[0])))
  
    i=0
    for z in json_load:
        i=i+1
        if i > 4: break
        if i==1: x=json_load[z]
        if i==2: y=json_load[z]
        if i==3: w=json_load[z]
        if i==4:h=json_load[z] 
       
    
    x=int(x)
    y=int(y)
    w=int(w)
    h=int(h)
    limitY=int(h/2)
    limitX=int(w/2)
    cropLicense=img[y-limitY:y+limitY,x-limitX:x+limitX]
    #cv2.imshow("Crop", cropLicense)
    #cv2.waitKey(0)
    TabcropLicense.append(cropLicense)
    return TabcropLicense, x, y, limitX, limitY


###########################################################
# MAIN
##########################################################

TabLicensesmax=[]
ContLicensesmax=[]
TimeIniLicensesmax=[]
TimeEndLicensesmax=[]

ContDetected=0
ContNoDetected=0
TotHits=0
TotFailures=0
ContSnapshots=0
with open( "VIDEOLicenseResults.txt" ,"w") as   w:
    
     cap = cv2.VideoCapture(dirVideo)
     # https://levelup.gitconnected.com/opencv-python-reading-and-writing-images-and-videos-ed01669c660c

     fourcc = cv2.VideoWriter_fourcc(*'MP4V')
     fps=5.0
     frame_width = 680
     frame_height = 480
     cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
     size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
     
     video_writer = cv2.VideoWriter('demonstration.mp4',fourcc,fps, size)
     while (cap.isOpened()):
     	ret, img = cap.read()
        
     
     	if ret != True:	break
        
     	else:
            #print(ret) 
     		#	cv2.imshow('original video', img)
     		#	
     		#	if cv2.waitKey(25) & 0xFF == ord('q'):
     		#		break     
            gray=img
            # speed up a little
            ContSnapshots=ContSnapshots+1
            if ContSnapshots < SpeedUpSnapshot:
                continue
            else:
                ContSnapshots=0
            
           
            License="License"
            TabImgSelect , x, y, limitX, limitY =DetectLicenseWithRoboflow(gray)
            if TabImgSelect==[]:
                print(License + " NON DETECTED")
                ContNoDetected=ContNoDetected+1 
                continue
            else:
                ContDetected=ContDetected+1
                print(License + " DETECTED ")
        
            gray=TabImgSelect[0]  
           
            x_off=3
            y_off=2
            
            x_resize=220
            y_resize=70
            
            Resize_xfactor=1.78
            Resize_yfactor=1.78
            
            ContLoop=0
            
            SwFounded=0
            
            BilateralOption=0
            
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
            #########################################
            #########################################
            print ("")  
            lineaw=[]
            lineaw.append(License) 
            lineaw.append(licensemax)
            lineaWrite =','.join(lineaw)
            lineaWrite=lineaWrite + "\n"
            w.write(lineaWrite)
            #if len(licensemax) < 2: continue
            SwFounded=0
            for i in range( len(TabLicensesmax)):
                if licensemax==TabLicensesmax[i]:
                    ContLicensesmax[i]=ContLicensesmax[i]+1
                    TimeEndLicensesmax[i]=time.time()
                    SwFounded=1
                    break
            if SwFounded ==0:    
                    TabLicensesmax.append(licensemax)
                    ContLicensesmax.append(1)
                    TimeIniLicensesmax.append(time.time())
                    TimeEndLicensesmax.append(time.time())
            #y-limitY:y+limitY,x-limitX:x+limitX 
            start_point=(x-limitX,y-limitY) 
            end_point=(x+limitX, y+limitY)
            color=(0,0,255)
            # Using cv2.rectangle() method
            # Draw a rectangle with blue line borders of thickness of 2 px
            img = cv2.rectangle(img, start_point, end_point,(36,255,12), 1)
            # Put text
            text_location = (x, y)
            text_color = (255,255,255)
            cv2.putText(img, licensemax ,text_location
                    , cv2.FONT_HERSHEY_SIMPLEX , 1
                    , text_color, 2 ,cv2.LINE_AA)
            cv2.imshow('Frame', img)
        # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'): break 
            # saving video
            video_writer.write(img)
           # a los 10 minutos = 600 segundos acaba     
            if time.time() - TimeIni > TimeLimit:
                
                break
               
                #print(TabLicensesmax)
                #print(ContLicensesmax)
                #break
     cap.release()
     video_writer.release()
     cv2.destroyAllWindows()
     w.close()
     with open( "VIDEOLicenseSummary.txt" ,"w") as   w1:
      for j in range (len(TabLicensesmax)):
          if ContLicensesmax[j] < LimitSnapshot:continue
          if TabLicensesmax[j] == "":continue
          Duration=TimeEndLicensesmax[j]-TimeIniLicensesmax[j]
          #Duration=Duration/ContLicensesmax[j]
          print(TabLicensesmax[j] + " snapshots: "+  str(ContLicensesmax[j]) + " Duration = "+str(Duration))
          lineaw1=[]
          lineaw1.append(TabLicensesmax[j]) 
          lineaw1.append(str(ContLicensesmax[j])) 
          lineaw1.append(str(Duration))
          lineaWrite =','.join(lineaw1)
          lineaWrite=lineaWrite + "\n"
          w1.write(lineaWrite)
     w1.close()
print("")           
