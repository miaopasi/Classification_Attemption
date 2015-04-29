__author__ = 'Xiaolong Shen sxl@nexdtech.com'

from numpy import *
from load_BLE_data import *
import numpy as np


def accuracy(res_cls, cls, cls_pos):
	RMSE = []
	for n in range(len(res_cls)):
		cls_res = res_cls[n]
		cls_tar = cls[n]
		pos_res = array(cls_pos[int(cls_res)])
		pos_tar = array(cls_pos[int(cls_tar)])
		pos_diff = pos_res
		se = np.sqrt(((pos_tar - pos_diff) ** 2).sum())
		RMSE.append(se)
	return RMSE


data = LoadData('./data','./rawdata/19/nexd.model.coord')
test_data = LoadData('./testdata', None, data.marker_dict)


"""
Test On NeuroLab.
As claimed PyBrain is out of maintained. Maybe catch up with cybrain or FANN Later.
Functionality Test First.
"""

import neurolab as nl
import numpy as np


# Get The Accuracy of Network
# RMSE = accuracy(NetRes, TarData, data.mat_res.cls_coord)
# print "++++++++++++++++++++++++++++++++++\n%s" % RMSE

def test_FANN():
	util = NeuralNetworkTest()
	train_data = data.mat_res.mat
	train_data = util.check_matrix(train_data)
	class_num = len(data.mat_res.sep_mat.keys())
	train_tar = util.build_tar_mat(class_num, data.mat_res.mat.shape[0], data.mat_res.cls)
	train_tar = util.check_matrix(train_tar)

	# Normalize input Data
	train_data = - train_data / 100.0

	# Check For The Correctness
	print train_data.shape
	print train_tar.shape

	data = libfann.training_data()
	data.set_train_data(train_data, train_tar)

	input_layer_num = data.num_input_train_data()
	output_layer_num = data.num_output_train_data()
	hidden_layer_num = int(float(input_layer_num + output_layer_num) / 2.0)

	net = libfann.neural_net()
	net.create_sparse_array(1, (input_layer_num, hidden_layer_num, output_layer_num))
	net.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC_STEPWISE)
	net.set_activation_function_output(libfann.SIGMOID_STEPWISE)

	net.train_on_data(data, 1000, 10, 0.01)

	net.save("BLE_Trained_Net_1000_0.01.conf")
	#
	res = zeros((test_data.mat_res.mat.shape[0], train_tar.shape[1]))
	for i, test_array in enumerate(test_data.mat_res.mat):
		res[i, :] = net.run(-test_array / 100.0)

	print res





def test_Neurolab(self):
	# Build Params For Network
	MIN = -100
	MAX = 0
	InputLayerSize = data.mat_res.mat.shape[1]

	ClassNum = len(data.mat_res.sep_mat.keys())

	# OutputLayerSize = len(data.mat_res.sep_mat.keys())
	OutputLayerSize = ClassNum
	HiddenLayerSize = int(float(InputLayerSize + OutputLayerSize) / 2.0)

	InputRangeParam = [[MIN, MAX] for i in range(InputLayerSize)]
	LayerNodes = [HiddenLayerSize, OutputLayerSize]

	# Extract Input And Output Of Network Training
	InputData = data.mat_res.mat
	# TarData = array(data.mat_res.cls).reshape(len(data.mat_res.cls), 1)
	# Get Target Data




	TarData = build_tar_mat(ClassNum, data.mat_res.mat.shape[0], data.mat_res.cls)

	# Extract Input And Default Output Of Network Test
	TestData = test_data.mat_res.mat
	TestTar = test_data.mat_res.cls

	# Create Network
	net = nl.net.newff(InputRangeParam, LayerNodes)
	print "Created Network Done"

	# Train network
	error = net.train(InputData, TarData, show=100)
	print "Network Training Done"

	# Simulate network
	NetRes = net.sim(InputData)
	print "Simulation of Network Done"

	print NetRes