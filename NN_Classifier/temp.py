from load_BLE_data import *
from numpy import *
from NN_Test_Class import *
from fann2 import libfann
import fann2
data = LoadData('../Data/data','../Data/rawdata/19/nexd.model.coord')
test_data = LoadData('../Data/testdata', None, data.marker_dict)

def get_weighted_pos(res, pos):
	res_pos = zeros(asarray(pos[0]).shape)
	for i in range(len(res)):
		res_pos += array(pos[i]) * res[i]
	res_pos = res_pos / res.sum()
	return res_pos



"""
Functions For Tidy Up Data
"""
#
util = NeuralNetworkTest()
# res = util.fann_ble_test_accum(data, test_data, True, "BLE_Network.conf")
res = util.fann_ble_test_recovered_accum(data, test_data, True, "BLE_Network.conf", 5)
# #
# res_cls = zeros((res.shape[0], 1))
# for i, arr in enumerate(res):
# 	mm = arr.max()
# 	ind = asarray(arr).argmax(0)
# 	res_cls[i] = ind

RMSE = util.weighted_accuracy(res, data.mat_res.cls, data.mat_res.cls_coord)
print RMSE
print mean(RMSE)

train_data = data.mat_res.mat
train_data = util.check_matrix(train_data)
class_num = len(data.mat_res.sep_mat.keys())
train_tar = util.build_tar_mat_ref(class_num, data.mat_res.mat.shape[0], data.mat_res.cls, 4)
train_tar = util.check_matrix(train_tar)
data = libfann.training_data()
data.set_train_data(train_data, train_tar)
util.net.reset_MSE()
util.net.test_data(data)

print util.net.get_MSE()