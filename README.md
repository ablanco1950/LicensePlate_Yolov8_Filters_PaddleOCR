# LicensePlate_Yolov8_Filters_PaddleOCR
Recognition of license plate numbers, in any format, by automatic detection with Yolov8, pipeline of filters and  paddleocr as OCR

The main improvement with respect to the project presented before ( https://github.com/ablanco1950/LicensePlate_Yolov8_MaxFilters) has been the use of paddleocr instead of pytesseract as well as the reduction of the number of filters, some of which although they were right in certain circumstances produced noise in other.

<video src='https://github.com/ablanco1950/LicensePlate_Yolov8_Filters_PaddleOCR/blob/main/demonstration1.mp4' width=180/>

Requirements:

paddleocr must be installed ( https://pypi.org/project/paddleocr/)

pip install paddleocr 

yolo must be installed, if not, follow the instructions indicated in:
  https://learnopencv.com/ultralytics-yolov8/#How-to-Use-YOLOv8?

pip install ultralytics

also must be installed the usual modules in computer vision:  numpy, cv2, os, re, imutils,  parabolic

Functioning:


Download the project to a folder on disk.

Download to that folder the roboflow files that will be used for training by yolov8:

https://public.roboflow.com/object-detection/license-plates-us-eu/3

In that folder you should find the train and valid folders necessary to build the model

To ensure the version, the used roboflow file roboflow.zip is attached

Unzip the file with the test images test6Training.zip, taking into account when unzipping you can create a folder
test6Training inside the test6Training folder,there should be only one test6Training folder, otherwise you will not find the
test images

Model Train:

the train and valid folders of the roboflow folder, resulting from the unziping of robflow.zip, must be placed in the same directory where the execution program LicensePlateYolov8Train.py is located, according to the requirements indicated in license_data.yaml

run the program

LicensePlateYolov8Train.py

which only has a few lines, but the line numbered 7 should indicate the full path where the license_data.yaml file is located.

Running from a simple laptop, the 100 epochs of the program will take a long time, but you can always lower the cover of the laptop and
continue the next day (besides, there are only 245 images for training). As obtaining best.pt is problematic, the one used in the project tests is attached.

As a result, inside the project folder, the directory runs\detect\trainN\weights( where in trainN, N indicates
 the last train directory created, in which the best.pt file is located), best.pt is the base of the model and
 is referenced in line 17 of the GetNumberInternationalLicensePlate_Yolov8_Filters_PaddleOCR.py program (modify the route, the name of trainN, so that it points to the last train and best.pt created

As obtaining best.pt is problematic, the one used in the project tests is attached,it must be  adjusted the route in instruction 17 in GetNumberInternationalLicensePlate_Yolov8_Filters_PaddleOCR.py

Run the program.

GetNumberInternationalLicensePlate_Yolov8_Filters_PaddleOCR.py

The car license plates and successes or failures through the different filters appear on the screen.

The LicenseResults.txt file lists the car license plates with their corresponding recognized ones.

In a test with 117 images, 100 hits are achieved

By changing the path in instruction 12, any other image directory can be tested (In this case, the LicenseResults.txt file must be consulted to indicate the license plates, since the files are not named with the license number, as in test6Training occurs, it cannot be determined if the assignment was successful automatically)

The video version is also included:

VIDEOGetNumberInternationalLicensePlate_Yolov8_Filters_PaddleOCR_Demonstration.py

operating on the attached video:

Traffic IP Camera video.mp4

downloaded from project:
https://github.com/anmspro/Traffic-Signal-Violation-Detection-System/tree/master/Resources

In its execution, on the monitor screen, the detected license plates are detailed with a summary at the end.

Three files are obtained:

VIDEOLicenseResults,txt with the registration of license plates detected with a lot of noise.
 
VIDEOLicenseSummary.txt with the following results, which seem pretty tight as can be seen visually from the video.

A8254S,145,198.2291808128357

AR606L,10,31.03719687461853

AE670S,10,15.752639770507812

A3K96,8,25.679476976394653

A3K961,3,14.658559083938599

A968B6,5,7.775115013122559

AV6190,10,17.38904595375061

The first field is the license plate detected and the second is the number of snapshots of that license plate.

As a maximum number of snapshots of 3 has been set (LimitSnapshot=3 parameter in the program), to avoid noise, the license plate of the APHI88 car that was going faster and that only recorded one snapshot does not appear (it can be checked in the VIDEOLicenseResults.txt logging file)

Also is produced a summary video: demonstration.mp4

Two videos of test results: demonstration1.mp4 and demonstration2.mp4 are attached.

The program is prepared to run in a time of 800 seconds (parameter: TimeLimit) so you have to wait that time until it ends or press q key.

More precise and exploitable results, although less apparent and more slowly, are obtained by executing:

VIDEOGetNumberInternationalLicensePlate_RoboflowModel_Filters_PaddleOCR.py

Other test videos can be downloaded from the addresses indicated in the program and in the references section.


References:

https://pypi.org/project/paddleocr/

https://learnopencv.com/ultralytics-yolov8/#How-to-Use-YOLOv8?

https://public.roboflow.com/object-detection/license-plates-us-eu/3

https://docs.ultralytics.com/python/

https://medium.com/@chanon.krittapholchai/build-object-detection-gui-with-yolov8-and-pysimplegui-76d5f5464d6c

https://medium.com/@alimustoofaa/how-to-load-model-yolov8-onnx-cv2-dnn-3e176cde16e6

https://medium.com/adevinta-tech-blog/text-in-image-2-0-improving-ocr-service-with-paddleocr-61614c886f93

https://machinelearningprojects.net/number-plate-detection-using-yolov7/

https://github.com/ablanco1950/LicensePlate_Yolov8_MaxFilters

Filters:

https://gist.github.com/endolith/334196bac1cac45a4893#

https://stackoverflow.com/questions/46084476/radon-transformation-in-python

https://gist.github.com/endolith/255291#file-parabolic-py

https://learnopencv.com/otsu-thresholding-with-opencv/ 

https://towardsdatascience.com/image-enhancement-techniques-using-opencv-and-python-9191d5c30d45

https://blog.katastros.com/a?ID=01800-4bf623a1-3917-4d54-9b6a-775331ebaf05

https://programmerclick.com/article/89421544914/

https://anishgupta1005.medium.com/building-an-optical-character-recognizer-in-python-bbd09edfe438

https://datasmarts.net/es/como-usar-el-detector-de-puntos-clave-mser-en-opencv/

https://felipemeganha.medium.com/detecting-handwriting-regions-with-opencv-and-python-ff0b1050aa4e

https://github.com/victorgzv/Lighting-correction-with-OpenCV

https://medium.com/@yyuanli19/using-mnist-to-visualize-basic-conv-filtering-95d24679643e

Projects with videos to download to test:

https://github.com/anmspro/Traffic-Signal-Violation-Detection-System/tree/master/Resources
"Traffic IP Camera video.mp4"

https://github.com/hasaan21/Car-Number-Plate-Recognition-Sysytem
"vid.mp4"

//www.pexels.com/video/video-of-famous-landmark-on-a-city-during-daytime-1721294/
"Pexels Videos 1721294.mp4"
