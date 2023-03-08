# LicensePlate_Yolov8_Filters_PaddleOCR
A 85% hit rate is achieved (99 hits in 117 images) in the recognition of license plate numbers, in any format, by automatic detection with Yolov8, pipeline of filters and  paddleocr as OCR

The main improvement with respect to the project presented before ( https://github.com/ablanco1950/LicensePlate_Yolov8_MaxFilters) has been the use of paddleocr instead of pytesseract as well as the reduction of the number of filters, some of which although they were right in certain circumstances produced noise in other.

Requirements:

paddleocr must be installed ( https://pypi.org/project/paddleocr/)

pip install paddleocr 
