# API to detect car in an Image
This API detects whether the input image has a car or not
  
## Overview
This API use CNN to predict whether input image contains a car or not.  
Prediction model is written and trained in Keras and API is written in Python Flask.
  
## Pre-requisite / Dependency
This project use OpenCV for image manipulation and Keras + Tensorflow for model training and Prediction.  
This project also use Flask for API.  
  
Other required packages are documented in [requirements.txt](https://github.com/shakeelansari63/car-detector-api/blob/main/requirements.txt)  
  
## Install Dependencies
```
python setup.py install
```  
  
## Start API Server
```
python api.py
```  
  
## Usage
Prediction of car in an image is 2 step process.  
1. Upload image file on server
2. Predict the image
  
### Upload Image
Make a POST API call to below address with multipart file parameter.
```
http://[host]:[port]/upload
```
  
Here is sample example from Insomnia  -  
![upload](https://github.com/shakeelansari63/car-detector-api/blob/main/screenshots/upload.png)  
  
### Predict
The image id for uploaded file which is returned in upload api will be used for prediction.  
  
Make get API call to below address with imgid.  
```
http://[host]:[port]/predict
```  
  
Here is sample example from Insomnia  -  
![predict](https://github.com/shakeelansari63/car-detector-api/blob/main/screenshots/predict.png)