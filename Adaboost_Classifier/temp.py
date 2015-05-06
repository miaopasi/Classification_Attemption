__author__ = 'Xiaolong Shen sxl@nexdtech.com'

from Adaboost_wifi_test import *

ada_clf = Adaboost_Wifi_Classification()
util = Adaboost_Wifi_Util()

dir_path = "/Users/admin/Code/ProjectCode/Algorithm/Classification_Attemption/Data/rawdata/XinZhongGuan"
test_dir_path = "/Users/admin/Code/ProjectCode/Algorithm/Classification_Attemption/Data/rawdata/XinZhongGuan_old"
train_save_path = "wifi_floor_train.npz"
test_save_path = "wifi_floor_test.npz"
net_path = "wifi_floor_xinzhongguan.conf"

# floor_data, wifi_bld_list, td = util.load_wifi_data_new(dir_path, train_save_path)

data = load(train_save_path)
floor_data = data['floor'].item()
wifi_bld_list = list(data['wifilist'])
td = data['entry_num']

clf = ada_clf.ada_wifi_train(floor_data, wifi_bld_list, td)

test_data, wifi_bld_list, td = util.load_wifi_data_new(test_dir_path, test_save_path, wifi_bld_list)
data = load(test_save_path)
test_data = data['floor'].item()

ada_clf.ada_wifi_validate(clf, test_data, wifi_bld_list, td)
