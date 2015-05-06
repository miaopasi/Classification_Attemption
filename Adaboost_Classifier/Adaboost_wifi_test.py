__author__ = 'Xiaolong Shen sxl@nexdtech.com'

"""
Will Scheduled Later than NN
"""

from Adaboost_Classification_Class import *
from loadfile import LoadWifiData


class Adaboost_Wifi_Util(ClassificationUtility):
	def __init__(self):
		ClassificationUtility.__init__(self)
		pass

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


class Adaboost_Wifi_Classification(AdaboostClassification):
	def __init__(self):
		AdaboostClassification.__init__(self)

	def ada_wifi_train(self, floor_data, wifi_bld_list, td):
		all_data_mat = zeros((td, len(wifi_bld_list)))
		all_data_cls = zeros((td, 1))

		pos_pt = 0
		cls_dict = {}

		for i, xx in enumerate(floor_data):
			d = floor_data[xx].wifi_bld_mat.shape[0]
			all_data_mat[pos_pt:pos_pt + d, :] = floor_data[xx].wifi_bld_mat
			all_data_cls[pos_pt:pos_pt + d, 0] = i
			pos_pt += d
			cls_dict[i] = xx
		all_data_cls = all_data_cls.ravel()
		self.clf = self.learn_clf(all_data_mat, all_data_cls, True)

	def ada_wifi_validate(self, clf, floor_data, wifi_bld_list, td):
		all_data_mat = zeros((td, len(wifi_bld_list)))
		all_data_cls = zeros((td, 1))

		pos_pt = 0
		cls_dict = {}

		for i, xx in enumerate(floor_data):
			d = floor_data[xx].wifi_bld_mat.shape[0]
			all_data_mat[pos_pt:pos_pt + d, :] = floor_data[xx].wifi_bld_mat
			all_data_cls[pos_pt:pos_pt + d, 0] = i
			pos_pt += d
			cls_dict[i] = xx
		all_data_cls = all_data_cls.ravel()
		res = clf.predict(all_data_mat)

		cor = sum(res == all_data_cls)
		print "Cor: %s, All: %s, Rate: %s" % (cor, len(res), cor / len(res))

