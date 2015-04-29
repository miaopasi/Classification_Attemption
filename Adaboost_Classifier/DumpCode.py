__author__ = 'admin'

#
# svm_clf = SVC()
# svm_clf.fit(data_mat, data.mat_res.cls)
# svc_st = time.time()
# res_cls_svm = svm_clf.predict(data_mat)
# svc_ed = time.time()
# res_svm_score = svm_clf.score(data_mat, data.mat_res.cls)
#
#
# print "Score: %s" % res_svm_score
# print "Time Cost: %s, Ave: %s" % (svc_ed-svc_st, (svc_ed-svc_st)/data_mat.shape[0])
# RMSE = accuracy(res_cls_svm, data.mat_res.cls, data.mat_res.cls_coord)
# print "RMSE: %s, Var: %s, MAX: %s" % (mean(RMSE), std(RMSE), max(RMSE))
# print RMSE
#
# svc_clf = AdaBoostClassifier(SVC(kernel='linear', probability=True),n_estimators=3)
# svc_clf.fit(data_mat, data.mat_res.cls)
# svc_st = time.time()
# res_cls_svm = svc_clf.predict(data_mat)
# svc_ed = time.time()
# res_svm_score = svc_clf.score(data_mat, data.mat_res.cls)
#
#
# print "Score: %s" % res_svm_score
# print "Time Cost: %s, Ave: %s" % (svc_ed-svc_st, (svc_ed-svc_st)/data_mat.shape[0])
# RMSE = accuracy(res_cls_svm, data.mat_res.cls, data.mat_res.cls_coord)
# print "RMSE: %s, Var: %s, MAX: %s" % (mean(RMSE), std(RMSE), max(RMSE))
# print RMSE
#
# clf_svm = AdaBoostClassifier(SVC(probability=True), n_estimators=50, learning_rate=1)
# clf_svm.fit(data_mat, data.mat_res.cls)
# res_cls_svm = clf_svm.predict(data_mat)
# res_svm_score = clf_svm.score(data_mat, data.mat_res.cls)





clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=200, learning_rate=1)

data_mat = data.mat_res.mat
data_mat = data_mat
print data_mat
clf.fit(data_mat, data.mat_res.cls)

st = time.time()
res_cls = clf.predict(data_mat)
ed = time.time()

res_comp = (res_cls == data.mat_res.cls)
res_score = clf.score(data_mat, data.mat_res.cls)

print "Score: %s" % res_score
print "Time Cost: %s, Ave: %s" % (ed-st, (ed-st)/data_mat.shape[0])
RMSE = accuracy(res_cls, data.mat_res.cls, data.mat_res.cls_coord)
print "RMSE: %s, Var: %s, MAX: %s" % (mean(RMSE), std(RMSE), max(RMSE))
print RMSE
#


#
test_data_mat = test_data.mat_res.mat
res_cls_svm = clf.predict(test_data_mat)
print res_cls_svm
RMSE = accuracy(res_cls_svm, test_data.mat_res.cls, data.mat_res.cls_coord)
print "RMSE: %s, Var: %s, MAX: %s" % (mean(RMSE), std(RMSE), max(RMSE))
print RMSE


test_data_mat_re = resample_data(test_data.mat_res.mat)
res_cls_svm = clf.predict(test_data_mat_re)
print res_cls_svm
RMSE = accuracy(res_cls_svm, test_data.mat_res.cls, data.mat_res.cls_coord)
print "RMSE: %s, Var: %s, MAX: %s" % (mean(RMSE), std(RMSE), max(RMSE))
print RMSE

#
# res_cls_svm = svc_clf.predict(test_data_mat)
# print res_cls_svm
# RMSE = accuracy(res_cls_svm, test_data.mat_res.cls, data.mat_res.cls_coord)
# print "RMSE: %s, Var: %s, MAX: %s" % (mean(RMSE), std(RMSE), max(RMSE))
# print RMSE




from numpy import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from load_BLE_data import *
import time
from math import *


def accuracy(res_cls,cls,cls_pos):
	RMSE = []
	for n in range(len(res_cls)):
		cls_res = res_cls[n]
		cls_tar = cls[n]
		pos_res = array(cls_pos[cls_res])
		pos_tar = array(cls_pos[cls_tar])
		pos_diff = pos_res
		se = sqrt(((pos_tar - pos_diff) ** 2).sum()) / 133.0
		RMSE.append(se)
	return RMSE


def stack_data(matrix):
	depth = matrix.shape[0]
	length = matrix.shape[1]
	# print "LENGTH: %s, DEPTH: %s" %(length, depth)
	arr = zeros((1, length))
	count = zeros((1, length))
	for i in range(depth):
		for j in range(length):
			if matrix[i, j] == 0:
				continue
			arr[0, j] += matrix[i, j]
			count[0, j] += 1
	for j in range(length):
		if not count[0, j]==0:
			arr[0, j] = float(arr[0, j]) / float(count[0, j])
	return arr

def resample_data(data_mat):
	print data_mat.shape
	re_data_mat = empty_like(data_mat)

	for i in range(data_mat.shape[0]):
		arr = stack_data(data_mat[max(0, i-1):i+1, :])
		re_data_mat[i, :] = arr[0]
	return re_data_mat
