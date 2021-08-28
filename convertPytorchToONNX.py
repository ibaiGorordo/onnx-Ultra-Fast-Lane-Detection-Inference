import cv2
import torch
import scipy.special
import numpy as np
import torchvision
from enum import Enum
import onnx
import onnxruntime

from ultrafastLaneDetector.model import parsingNet

class ModelType(Enum):
	TUSIMPLE = 0
	CULANE = 1

class ModelConfig():

	def __init__(self, model_type):

		if model_type == ModelType.TUSIMPLE:
			self.init_tusimple_config()
		else:
			self.init_culane_config()

	def init_tusimple_config(self):
		self.img_w = 1280
		self.img_h = 720
		self.griding_num = 100
		self.cls_num_per_lane = 56

	def init_culane_config(self):
		self.img_w = 1640
		self.img_h = 590
		self.griding_num = 200
		self.cls_num_per_lane = 18

def convert_model(model_path, onnx_file_path, model_type=ModelType.TUSIMPLE):

	# Load model configuration based on the model type
	cfg = ModelConfig(model_type)


	# Load the model architecture
	net = parsingNet(pretrained = False, backbone='18', cls_dim = (cfg.griding_num+1,cfg.cls_num_per_lane,4),
					use_aux=False) # we dont need auxiliary segmentation in testing


	state_dict = torch.load(model_path, map_location='cpu')['model'] # CPU

	compatible_state_dict = {}
	for k, v in state_dict.items():
		if 'module.' in k:
			compatible_state_dict[k[7:]] = v
		else:
			compatible_state_dict[k] = v

	# Load the weights into the model
	net.load_state_dict(compatible_state_dict, strict=False)

	img = torch.zeros(1, 3, 288, 800).to('cpu')
	torch.onnx.export(net, img, onnx_file_path, verbose=True)

	model = onnx.load(onnx_file_path)

	# Check that the IR is well formed
	onnx.checker.check_model(model)
	# Print a human readable representation of the graph
	print("==============================================================================================")

if __name__ == '__main__':
	
	onnx_model_path = "tusimple_18.onnx"
	model_path = "models/tusimple_18.pth"
	model_type = ModelType.TUSIMPLE

	convert_model(model_path, onnx_model_path, model_type)



