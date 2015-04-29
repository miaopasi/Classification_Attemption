__author__ = 'Xiaolong Shen sxl@nexdtech.com'

from numpy import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from load_BLE_data import *
import time
from math import *


class ClassificationUtility:
	def __init__(self):
		pass

	def accuracy(self, res_cls, cls, cls_pos):
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

	def stack_data(self, mat):
		depth = mat.shape[0]
		length = mat.shape[1]
		# print "LENGTH: %s, DEPTH: %s" %(length, depth)
		arr = zeros((1, length))
		count = zeros((1, length))
		for i in range(depth):
			for j in range(length):
				if mat[i, j] == 0:
					continue
				arr[0, j] += mat[i, j]
				count[0, j] += 1
		for j in range(length):
			if not count[0, j] == 0:
				arr[0, j] = float(arr[0, j]) / float(count[0, j])
		return arr

	def resample_data(self, data_mat):
		print data_mat.shape
		re_data_mat = empty_like(data_mat)

		for i in range(data_mat.shape[0]):
			arr = self.stack_data(data_mat[max(0, i - 1):i + 1, :])
			re_data_mat[i, :] = arr[0]
		return re_data_mat


class AdaboostClassification:
	def __init__(self):
		self.util = ClassificationUtility()
		self.DT_depth = 10
		self.num_estimator = 200
		self.learning_rate = 1

	def init_classifier(self):
		clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=self.DT_depth), n_estimators=self.num_estimator,
		                         learning_rate=self.learning_rate)
		return clf

	@classmethod
	def redef_classifier(cls, base_classifier, num_estimator, learning_rate):
		clf = AdaBoostClassifier(base_classifier, n_estimators=num_estimator, learning_rate=learning_rate)
		return clf

	@classmethod
	def train_clf(cls, clf, train_data, train_tar, display=True):
		clf.fit(train_data, train_tar)
		if display:
			st = time.time()
			clf.predict(train_data)
			ed = time.time()

			res_score = clf.score(train_data, train_tar)

			print "Score: %s" % res_score
			print "Time Cost: %s, Ave: %s" % (ed-st, (ed-st)/train_data.shape[0])
		return clf

	def test_clf(self, clf, test_data):
		pass

	def validate_clf(self, clf, test_data, test_tar, raw_coord, display=True):
		res = clf.predict(test_data)
		RMSE = self.util.accuracy(res, test_tar, raw_coord)
		if display:
			print "RMSE: %s, Var: %s, MAX: %s" % (mean(RMSE), std(RMSE), max(RMSE))
			print RMSE
		return res, RMSE

	def learn_clf(self, train_data, train_tar, display=True):
		clf = self.init_classifier()
		clf = self.train_clf(clf, train_data, train_tar, display)
		return clf

	def run(self, train_data, test_data, display=True):
		train_input = train_data.mat_res.mat
		train_tar = train_data.mat_res.cls
		test_input = test_data.mat_res.mat
		test_tar = test_data.mat_res.cls
		raw_coord = train_data.mat_res.cls_coord

		clf = self.learn_clf(train_input, train_tar, display)
		res, RMSE = self.validate_clf(clf, test_input, test_tar, raw_coord, display)

		return clf, res, RMSE

	def run_with_resample(self, train_data, test_data, display=True):
		train_input = train_data.mat_res.mat
		train_tar = train_data.mat_res.cls
		test_input = test_data.mat_res.mat
		test_tar = test_data.mat_res.cls
		raw_coord = train_data.mat_res.cls_coord

		# Resample Data to Acuumulate data

		test_input = self.util.resample_data(test_input)

		clf = self.learn_clf(train_input, train_tar, display)
		res, RMSE = self.validate_clf(clf, test_input, test_tar, raw_coord, display)

		return clf, res, RMSE






