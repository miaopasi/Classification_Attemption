__author__ = 'Xiaolong Shen sxl@nexdtech.com'
from NN_Test_Class import *
from loadfile import LoadWifiData
import time

class NeuralNetworkWifiTest(NeuralNetworkTest):
	def __init__(self):
		NeuralNetworkTest.__init__(self)
		self.load_in = LoadWifiData()
		pass

	def load_wifi_data_old(self, dir_path, save_path="wifi_floor.npz"):
		file_list = os.listdir(dir_path)
		floor_data = {}
		for file_name in file_list:
			if file_name.endswith("wp"):
				name, ext = os.path.splitext(file_name)
				print name
				wifi_file = name + ".wifi"
				wifi_file = os.path.join(dir_path, wifi_file)
				wp_file = os.path.join(dir_path, file_name)
				floor_data[name] = self.load_in.extract(wp_file, wifi_file)
				# temp = load_in.extract_raw(wp_file, wifi_file)
				# print temp

		# Get All WiFi Listed
		t_bind = []
		for x in floor_data:
			t_bind += floor_data[x].wifi_list
		wifi_bld_list = list(set(t_bind))

		TD = 0

		for x in floor_data:
			D = floor_data[x].wifi_matrix.shape[0]
			TD += D
			L = len(wifi_bld_list)
			t_mat = zeros((D, L))
			for i, j in enumerate(floor_data[x].wifi_list):
				ind = wifi_bld_list.index(j)
				t_mat[:, ind] = floor_data[x].wifi_matrix[:, i]
			floor_data[x].wifi_bld_mat = t_mat

		savez(save_path, floor=floor_data, wifilist=wifi_bld_list, entry_num=TD)

		return floor_data, wifi_bld_list, TD

	def load_wifi_data_new(self, dir_path, save_path="wifi_floor_temp.npz", ref_list=[]):
		load_in = LoadWifiData()
		file_list = os.listdir(dir_path)
		floor_data = {}
		for floor in file_list:
			print floor
			fl_path = os.path.join(dir_path, floor)
			if os.path.isfile(fl_path):
				continue
			floor_file_list = os.listdir(fl_path)
			for file_name in floor_file_list:
				if file_name.endswith("wifi"):
					name, ext = os.path.splitext(file_name)
					print name
					wifi_file = name + ".wifi"
					wifi_file = os.path.join(fl_path, wifi_file)
					floor_data[floor] = load_in.extract_wifi(wifi_file, ref_list)
		t_bind = []
		for x in floor_data:
			t_bind += floor_data[x].wifi_list
		wifi_bld_list = list(set(t_bind))

		TD = 0

		for x in floor_data:
			D = floor_data[x].wifi_matrix.shape[0]
			TD += D
			L = len(wifi_bld_list)
			t_mat = zeros((D, L))
			for i, j in enumerate(floor_data[x].wifi_list):
				ind = wifi_bld_list.index(j)
				t_mat[:, ind] = floor_data[x].wifi_matrix[:, i]
			floor_data[x].wifi_bld_mat = t_mat

		savez(save_path, floor=floor_data, wifilist=wifi_bld_list, entry_num=TD)
		return floor_data, wifi_bld_list, TD

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

	def fann_wifi_train(self, floor_data, wifi_bld_list, td, save_path="wifi_floor.conf"):
		all_data_mat = zeros((td, len(wifi_bld_list)))
		all_data_cls = zeros((td, len(floor_data.keys())))

		pos_pt = 0
		cls_dict = {}

		for i, xx in enumerate(floor_data):
			d = floor_data[xx].wifi_bld_mat.shape[0]
			all_data_mat[pos_pt:pos_pt + d, :] = floor_data[xx].wifi_bld_mat
			all_data_cls[pos_pt:pos_pt + d, i] = 1
			pos_pt += d
			cls_dict[i] = xx
		net = self.fann_learn_save_norm(all_data_mat, all_data_cls, save_path, False)
		self.net = net
		return net

	def fann_wifi_test(self, floor_data, rec_path="wifi_floor.conf"):
		cls_dict = {}
		for i, xx in enumerate(floor_data):
			cls_dict[i] = xx

		class_num = len(floor_data.keys())
		for x in cls_dict:
			floor = cls_dict[x]
			st = time.time()
			res = self.fann_wifi_test_recovered(class_num, floor_data[floor].wifi_bld_mat, False, rec_path)
			ed = time.time()
			print res
			print "Time Comsumption: %s , Ave Time Consumption: %s" % (ed-st, (ed-st) / floor_data['F1'].wifi_bld_mat.shape[0])
			res_cls = zeros((res.shape[0],1))
			cor_count = 0;
			for i, arr in enumerate(res):
				res_cls[i] = arr.argmax()
				if res_cls[i] == x:
					cor_count += 1

			print "Res_Cor: %s,Total Amount : %s, Correct Rate: %s" % (cor_count, len(res_cls), cor_count / float(len(res_cls)))
		#

