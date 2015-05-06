from loadfile import LoadWifiData
import os
from numpy import *
from NN_WiFi_Test_Class import *
#
util = NeuralNetworkWifiTest()

dir_path = "/Users/admin/Code/ProjectCode/Algorithm/Classification_Attemption/Data/rawdata/XinZhongGuan"
test_dir_path = "/Users/admin/Code/ProjectCode/Algorithm/Classification_Attemption/Data/rawdata/XinZhongGuan_old"
train_save_path = "wifi_floor_train.npz"
test_save_path = "wifi_floor_test.npz"
net_path = "wifi_floor_xinzhongguan.conf"

# floor_data, wifi_bld_list, td = util.load_wifi_data_new(dir_path, train_save_path)
# net = util.fann_wifi_train(floor_data, wifi_bld_list, td, net_path)
data = load(train_save_path)
floor_data = data['floor'].item()
wifi_bld_list = list(data['wifilist'])
td = data['entry_num']

# test_data, wifi_bld_list, td = util.load_wifi_data_new(test_dir_path, test_save_path, wifi_bld_list)
data = load(test_save_path)
test_data = data['floor'].item()
util.fann_wifi_test(test_data, net_path)



# load_in = LoadWifiData()
#
#
# #
# ori_path = "/Users/admin/Code/ProjectCode/Algorithm/Classification_Attemption/Data/wifidata"
# save_path = "wifi_floor.npz"
# # floor_data, wifi_bld_list, TD = util.load_wifi_data_old(ori_path)
#
# data = load(save_path)
# floor_data = data['floor'].item()
# wifi_bld_list = list(data['wifilist'])
# TD = data['entry_num']
#
# # net, cls_dict = util.fann_wifi_train(floor_data, wifi_bld_list, TD)
#
#
# util.fann_wifi_test(floor_data)



