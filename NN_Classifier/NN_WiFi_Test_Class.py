__author__ = 'Xiaolong Shen sxl@nexdtech.com'
from NN_Test_Class import *


class NeuralNetworkWifiTest(NeuralNetworkTest):
	def __init__(self):
		pass

	def fann_wifi_test_recovered(self, class_num, test_data, normalize=True, savepath="./temp_save.conf"):
		if not os.path.exists(savepath):
			print "No File Included"
			return []
		net = libfann.neural_net()
		net.create_from_file(savepath)
		res = zeros((test_data.shape[0], class_num))
		for i, test_array in enumerate(test_data):
			if normalize:
				# test_array = -test_array / 100.0
				test_array = self.normalize(test_array, self.m_min, self.m_max)
			res[i, :] = net.run(test_array)
		self.net = net
		return res

	def fann_wifi_test_recovered_accum(self, class_num, test_data, normalize=True, savepath="./temp_save.conf",\
	                                   accum_depth=3):
		if not os.path.exists(savepath):
			print "No File Included"
			return []
		net = libfann.neural_net()
		net.create_from_file(savepath)
		res = zeros((test_data.shape[0], class_num))
		for i, test_array in enumerate(test_data):
			test_array_raw = test_data[max(0, i - accum_depth):i + 1, :]
			test_array = self.stack_data(test_array_raw)[0]
			if normalize:
				# test_array = -test_array / 100.0
				test_array = self.normalize(test_array, self.m_min, self.m_max)
			# print test_array
			res[i, :] = net.run(test_array.round())
		self.net = net
		return res