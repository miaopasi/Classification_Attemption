from loadfile import LoadWifiData
import os
from numpy import *
from NN_Test_Class import *

util = NeuralNetworkTest()
load_in = LoadWifiData()


#
# ori_path = "/Users/admin/Code/ProjectCode/Algorithm/Classification_Attemption/Data/wifidata"
# file_list = os.listdir(ori_path)
# floor_data = {}
# for file_name in file_list:
# 	if file_name.endswith("wp"):
# 		name, ext = os.path.splitext(file_name)
# 		print name
# 		wifi_file = name + ".wifi"
# 		wifi_file = os.path.join(ori_path, wifi_file)
# 		wp_file = os.path.join(ori_path, file_name)
# 		floor_data[name] = load_in.extract(wp_file, wifi_file)
#
# # Get All WiFi Listed
# t_bind = []
# for x in floor_data:
# 	t_bind += floor_data[x].wifi_list
# wifi_bld_list = list(set(t_bind))
#
# TD = 0
#
# for x in floor_data:
# 	D = floor_data[x].wifi_matrix.shape[0]
# 	TD += D
# 	L = len(wifi_bld_list)
# 	t_mat = zeros((D, L))
# 	for i, j in enumerate(floor_data[x].wifi_list):
# 		ind = wifi_bld_list.index(j)
# 		t_mat[:, ind] = floor_data[x].wifi_matrix[:, i]
# 	floor_data[x].wifi_bld_mat = t_mat
#
# for x in floor_data:
# 	print floor_data[x].wifi_bld_mat
#
# savez("wifi_floor.npz", floor=floor_data, wifilist=wifi_bld_list, entry_num=TD)


data = load('wifi_floor.npz')
floor_data = data['floor'].item()
wifi_bld_list = list(data['wifilist'])
TD = data['entry_num']
# print floor_data['F1']
print len(floor_data.keys())
print len(wifi_bld_list)
#
all_data_mat = zeros((TD, len(wifi_bld_list)))
all_data_cls = zeros((TD, len(floor_data.keys())))

pos_pt = 0
cls_dict = {}
#
for i, x in enumerate(floor_data):
	D = floor_data[x].wifi_bld_mat.shape[0]
	all_data_mat[pos_pt:pos_pt+D, :] = floor_data[x].wifi_bld_mat
	all_data_cls[pos_pt:pos_pt+D, i] = 1
	pos_pt += D
	cls_dict[i] = x
#
# print all_data_cls
#
# net = util.fann_learn_save(all_data_mat, all_data_cls,  "wifi_floor.conf")

print cls_dict

import time
class_num = len(floor_data.keys())
for x in cls_dict:
	floor = cls_dict[x]
	st = time.time()
	res = util.fann_wifi_test_recovered_accum(class_num, floor_data[floor].wifi_bld_mat, False, 'wifi_floor.conf', 5)
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





