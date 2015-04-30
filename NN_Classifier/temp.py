from loadfile import LoadData
import os
from numpy import *


load_in = LoadData()

ori_path = "/Users/admin/Code/ProjectCode/Algorithm/Classification_Attemption/Data/wifidata"
file_list = os.listdir(ori_path)
floor_data = {}
for file_name in file_list:
	if file_name.endswith("wp"):
		name, ext = os.path.splitext(file_name)
		print name
		wifi_file = name + ".wifi"
		wifi_file = os.path.join(ori_path, wifi_file)
		wp_file = os.path.join(ori_path, file_name)
		floor_data[name] = load_in.extract(wp_file, wifi_file)

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

for x in floor_data:
	print floor_data[x].wifi_bld_mat

all_data_mat = zeros((TD, len(wifi_bld_list)))
all_data_cls = zeros((TD,1))
cls_pt = 1
pos_pt = 0
for x in floor_data:
	D = floor_data[x].wifi_bld_mat.shape[0]
	all_data_mat[pos_pt:pos_pt+D, :] = floor_data[x].wifi_bld_mat
	pos_pt += D
	print "pos_pt : %s , pos_pt+D : %s" %(pos_pt, pos_pt+D)
	all_data_cls[pos_pt:pos_pt+D, :] = cls_pt
	cls_pt += 1

print all_data_cls



