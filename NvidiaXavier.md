# onnx-Ultra-Fast-Lane-Detection-Inference With Nvidia Xavier
 Example scripts for the detection of lanes using the [ultra fast lane detection model](https://github.com/cfzd/Ultra-Fast-Lane-Detection) in ONNX.

![!Ultra fast lane detection](https://github.com/ibaiGorordo/onnx-Ultra-Fast-Lane-Detection-Inference/blob/main/doc/img/detected%20lanes.jpg)
Source: https://www.flickr.com/photos/32413914@N00/1475776461/

# Pytorch inference
For performing the inference in Pytorch, check my other repository **[Ultrafast Lane Detection Inference Pytorch](https://github.com/ibaiGorordo/Ultrafast-Lane-Detection-Inference-Pytorch-)**.

# Requirements

 * **OpenCV**, **scipy**, **onnx** and **onnxruntime**. **pafy**  and **youtube-dl**, **Nvidia Xavier AGX**, **JetPack 4.6** and **Python3.6**  are required for youtube video inference. 

# Installation

```
pip3 install opencv-python
pip3 install scipy
Download >> https://nvidia.app.box.com/s/bfs688apyvor4eo8sf3y1oqtnarwafww
Install  >> pip3 install onnxruntime_gpu-1.8.0-cp36-cp36m-linux_aarch64.whl
pip3 install scikit-build
if you found not match version try to upgrade the PIP >> sudo -H pip3 install --upgrade pip

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
 
## Result
- YoutTube: https://www.youtube.com/watch?v=yBmcYDke7Wg
 
 Original video: https://youtu.be/2CIxM7x-Clc (by Yunfei Guo)
 
