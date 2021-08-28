# onnx-Ultra-Fast-Lane-Detection-Inference
 Example scripts for the detection of lanes using the [ultra fast lane detection model](https://github.com/cfzd/Ultra-Fast-Lane-Detection) in ONNX.

![!Ultra fast lane detection](https://github.com/ibaiGorordo/onnx-Ultra-Fast-Lane-Detection-Inference/blob/main/doc/img/detected%20lanes.jpg)
Source: https://www.flickr.com/photos/32413914@N00/1475776461/

# Pytorch inference
For performing the inference in Pytorch, check my other repository **[Ultrafast Lane Detection Inference Pytorch](https://github.com/ibaiGorordo/Ultrafast-Lane-Detection-Inference-Pytorch-)**.

# Requirements

 * **OpenCV**, **scipy**, **onnx** and **onnxruntime**. **pafy** and **youtube-dl** are required for youtube video inference. 
 
# Installation
```
pip install -r requirements.txt
pip install pafy youtube-dl
```

# ONNX model
The original model was converted to different formats (including .onnx) by [PINTO0309](https://github.com/PINTO0309), download the models from [his repository](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/140_Ultra-Fast-Lane-Detection) and save it into the **[models](https://github.com/ibaiGorordo/TfLite-Ultra-Fast-Lane-Detection-Inference/tree/main/models)** folder. 

ONNX Conversion script: https://github.com/cfzd/Ultra-Fast-Lane-Detection/issues/218

# Original Pytorch model
The pretrained Pytorch model was taken from the [original repository](https://github.com/cfzd/Ultra-Fast-Lane-Detection).

# Model info ([link](https://github.com/cfzd/Ultra-Fast-Lane-Detection))

 * **Input**: RGB image of size 800 x 200 pixels.
 * **Output**: Keypoints for a maximum of 4 lanes (left-most lane, left lane, right lane, and right-most lane).
 
# Examples

 * **Image inference**:
 
 ```
 python imageLaneDetection.py 
 ```
 
  * **Webcam inference**:
 
 ```
 python webcamLaneDetection.py
 ```
 
  * **Video inference**:
 
 ```
 python videoLaneDetection.py
 ```
 
 # [Inference video Example](https://youtu.be/0Owf6gef1Ew) 
 ![!Ultrafast lane detection on video](https://github.com/ibaiGorordo/Ultrafast-Lane-Detection-Inference-Pytorch-/blob/main/doc/img/laneDetection.gif)
 
 Original video: https://youtu.be/2CIxM7x-Clc (by Yunfei Guo)
 
