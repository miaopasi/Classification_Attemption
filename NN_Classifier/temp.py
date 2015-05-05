from loadfile import LoadWifiData
import os
from numpy import *
from NN_WiFi_Test_Class import *

util = NeuralNetworkWifiTest()
load_in = LoadWifiData()


#
ori_path = "/Users/admin/Code/ProjectCode/Algorithm/Classification_Attemption/Data/wifidata"
save_path = "wifi_floor.npz"
floor_data, wifi_bld_list, TD = util.load_wifi_data_old(ori_path)

data = load(save_path)
floor_data = data['floor'].item()
wifi_bld_list = list(data['wifilist'])
TD = data['entry_num']

# #
# # print all_data_cls
# #
# net = util.fann_learn_save_norm(all_data_mat, all_data_cls, "wifi_floor_norm.conf")
#
# print cls_dict
#

net, cls_dict = util.fann_wifi_train(floor_data, wifi_bld_list, TD)




import time
class_num = len(floor_data.keys())
for x in cls_dict:
	floor = cls_dict[x]
	st = time.time()
	res = util.fann_wifi_test_recovered_accum(class_num, floor_data[floor].wifi_bld_mat, False, 'wifi_floor.conf', 10)
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





# net = util.net
# test_arr_raw = floor_data[floor].wifi_bld_mat[0:5, :]
# test_arr = util.stack_data(test_arr_raw)
# test_arr_single = floor_data[floor].wifi_bld_mat[5, :]
# print "TEST_ARR"
# print test_arr
# print "TEST_ARR_SINGLE"
# print test_arr_single
# res1 = net.run(test_arr[0])
# res2 = net.run(test_arr_single)
#
# print "RES1"
# print res1
# print "RES2"
# print res2
#

