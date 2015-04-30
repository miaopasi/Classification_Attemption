__author__ = 'Xiaolong Shen sxl@nexdtech.com'
from numpy import *
from load_BLE_data import *
import numpy
from fann2 import libfann
import os


class NeuralNetworkTest:
	def __init__(self):
		self.map_ratio = 100.0
		self.m_min = -90.0
		self.m_max = -20.0
		pass

	def update_data(self, train_data, test_data, train_tar):
		self.train_data = train_data
		self.test_data = test_data
		self.train_tar = train_tar

	def build_tar_mat(self, class_num, inputlayersize, cls):
		TarData = zeros((inputlayersize, class_num))
		for n in range(len(cls)):
			ind = cls[n]
			TarData[n, ind] = 1
		return TarData

	def build_tar_mat_ref(self, class_num, inputlayersize, cls, repeat_ref):
		real_class_num = int(class_num / repeat_ref)
		TarData = zeros((inputlayersize, real_class_num))
		for n in range(len(cls)):
			ind = numpy.mod(cls[n], real_class_num)
			TarData[n, ind] = 1
		return TarData

	def accuracy(self, res_cls, cls, cls_pos):
		RMSE = []
		for n in range(len(res_cls)):
			cls_res = res_cls[n]
			cls_tar = cls[n]
			pos_res = array(cls_pos[int(cls_res)])
			pos_tar = array(cls_pos[int(cls_tar)]) / self.map_ratio
			pos_diff = pos_res / self.map_ratio
			se = numpy.sqrt(((pos_tar - pos_diff) ** 2).sum())
			RMSE.append(se)
		return RMSE

	def get_weighted_pos(self, res, pos):
		res_pos = zeros(asarray(pos[0]).shape)
		add_w = 0.0
		for i in range(len(res)):
			if res[i] > 0.4:
				add_w += res[i]
				res_pos += array(pos[i]) * res[i]
		res_pos = res_pos / res.sum()
		return res_pos

	def weighted_accuracy(self, res, cls, cls_pos):
		RMSE = []
		for i, arr in enumerate(res):
			pos_res = self.get_weighted_pos(arr, cls_pos)
			cls_tar = cls[i]
			pos_tar = array(cls_pos[int(cls_tar)])
			se = numpy.sqrt(((pos_tar - pos_res) ** 2).sum()) / self.map_ratio
			RMSE.append(se)
		return RMSE

	def stack_data(self, matrix):
		matrix = self.check_matrix(matrix)
		depth = matrix.shape[0]
		length = matrix.shape[1]
		# print "LENGTH: %s, DEPTH: %s" %(length, depth)
		array = zeros((1, length))
		count = zeros((1, length))
		for i in range(depth):
			for j in range(length):
				if matrix[i, j] == 0:
					continue
				array[0, j] += matrix[i, j]
				count[0, j] += 1
		for j in range(length):
			if not count[0, j]==0:
				array[0, j] = float(array[0, j]) / float(count[0, j])
		return array


	# convinience method for 1-dimensional arrays
	# fann cannot handle those and leaves with a segfault :(
	def check_matrix(self, matrix):
		if matrix.ndim == 1:
			new = numpy.empty((matrix.shape[0], 1))

			for i, x in enumerate(matrix):
				new[(i, 0)] = x

			return new

		return matrix

	def normalize(self, matrix, m_min, m_max):
		empty_ind = where(matrix == 0)
		matrix[where(matrix < m_min)] = m_min
		matrix[where(matrix > m_max)] = m_max
		matrix = (matrix - m_min) / (m_max - m_min)
		matrix[empty_ind] = -0.1
		return matrix


	"""
	THIS IS THE FANN TEST PART CLEANED UP FOR EASIER CALLING
	"""
	def fann_init_net(self, hidden_func=libfann.SIGMOID_SYMMETRIC_STEPWISE, output_func=libfann.SIGMOID_STEPWISE):
		net = libfann.neural_net()
		net.set_activation_function_hidden(hidden_func)
		net.set_activation_function_output(output_func)
		return net

	def fann_train(self, train_data, train_tar, net):
		data = libfann.training_data()
		data.set_train_data(train_data, train_tar)
		input_layer_num = data.num_input_train_data()
		output_layer_num = data.num_output_train_data()
		hidden_layer_num = int(float(input_layer_num + output_layer_num) / 2.0)
		print "Training Network With Params:\nInput: %s, Hidden: %s, Output: %s" %(input_layer_num, hidden_layer_num, output_layer_num)
		net.create_sparse_array(1, (input_layer_num, hidden_layer_num, output_layer_num))
		net.train_on_data(data, 8000, 10, 0.00001)
		return net

	def fann_train_save(self, train_data, train_tar, net, savepath="./temp_save.conf"):
		net = self.fann_train(train_data, train_tar, net)
		net.save(savepath)
		return net

	def fann_learn(self, train_data, train_tar):
		net = self.fann_init_net()
		net = self.fann_train(train_data, train_tar, net)
		return net

	def fann_learn_save(self, train_data, train_tar, savepath="./temp_save.conf"):
		net = self.fann_learn(train_data, train_tar)
		net.save(savepath)
		return net

	# Cleaned Up Fast Build Of Network For Test With BLE Data
	def fann_ble_test(self, data, test_data, normalize=True, savepath="./temp_save.conf"):
		train_data = data.mat_res.mat
		train_data = self.check_matrix(train_data)
		class_num = len(data.mat_res.sep_mat.keys())
		train_tar = self.build_tar_mat_ref(class_num, data.mat_res.mat.shape[0], data.mat_res.cls, 4.0)
		train_tar = self.check_matrix(train_tar)
		if normalize:
			# train_data = - train_data / 100.0
			train_data = self.normalize(train_data, self.m_min, self.m_max)
		net = self.fann_learn_save(train_data, train_tar, savepath)
		res = zeros((test_data.mat_res.mat.shape[0], train_tar.shape[1]))
		for i, test_array in enumerate(test_data.mat_res.mat):
			if normalize:
				# test_array = -test_array / 100.0
				test_array = self.normalize(test_array, self.m_min, self.m_max)
			res[i, :] = net.run(test_array)
		self.net = net
		return res

	def fann_ble_test_recovered(self, data, test_data, normalize=True, savepath="./temp_save.conf"):
		if not os.path.exists(savepath):
			print "No File Included"
			return []
		net = libfann.neural_net()
		net.create_from_file(savepath)
		res = zeros((test_data.mat_res.mat.shape[0], len(data.mat_res.sep_mat.keys())))
		for i, test_array in enumerate(test_data.mat_res.mat):
			if normalize:
				# test_array = -test_array / 100.0
				test_array = self.normalize(test_array, self.m_min, self.m_max)
			res[i, :] = net.run(test_array)
		self.net = net
		return res

	def fann_ble_test_accum(self, data, test_data, normalize=True, savepath="./temp_save.conf", accum_depth=3):
		train_data = data.mat_res.mat
		train_data = self.check_matrix(train_data)
		class_num = len(data.mat_res.sep_mat.keys())
		train_tar = self.build_tar_mat_ref(class_num, data.mat_res.mat.shape[0], data.mat_res.cls, 4.0)
		train_tar = self.check_matrix(train_tar)
		if normalize:
			# train_data = - train_data / 100.0
			train_data = self.normalize(train_data, self.m_min, self.m_max)
		net = self.fann_learn_save(train_data, train_tar, savepath)
		res = zeros((test_data.mat_res.mat.shape[0], train_tar.shape[1]))
		for i, test_array in enumerate(test_data.mat_res.mat):
			test_array_raw = test_data.mat_res.mat[max(0, i - accum_depth):i+1, :]
			if normalize:
				# test_array = -test_array / 100.0
				test_array_raw = self.normalize(test_array_raw, self.m_min, self.m_max)
				test_array = test_array_raw.mean(axis=1)
			res[i, :] = net.run(test_array)
		self.net = net
		return res

	def fann_ble_test_recovered_accum(self, data, test_data, normalize=True, savepath="./temp_save.conf", accum_depth=3):
		if not os.path.exists(savepath):
			print "No File Included"
			return []
		net = libfann.neural_net()
		net.create_from_file(savepath)
		res = zeros((test_data.mat_res.mat.shape[0], int(len(data.mat_res.sep_mat.keys())/4.0)))
		for i, test_array_miao in enumerate(test_data.mat_res.mat):
			test_array_raw = test_data.mat_res.mat[max(0, i - accum_depth):i+1, :]
			test_array = self.stack_data(test_array_raw)
			if normalize:
				test_array = self.normalize(test_array, self.m_min, self.m_max)
			# print test_array
			res[i, :] = net.run(test_array)
		self.net = net
		return res


