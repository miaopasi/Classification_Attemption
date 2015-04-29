__author__ = 'Xiaolong Shen sxl@nexdtech.com'
from numpy import *
from load_BLE_data import *
from Adaboost_Classification_Class import *


data = LoadData('../Data/data', '../Data/rawdata/19/nexd.model.coord')
test_data = LoadData('../Data/testdata', None, data.marker_dict)

adatest = AdaboostClassification()
clf, res, RMSE = adatest.run(data, test_data)
