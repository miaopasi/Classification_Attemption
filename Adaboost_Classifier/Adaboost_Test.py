__author__ = 'Xiaolong Shen sxl@nexdtech.com'
from numpy import *
from load_BLE_data import *
from Adaboost_Classification_Class import *
import matplotlib.pyplot as plt
import matplotlib
data = LoadData('../Data/data', '../Data/rawdata/19/nexd.model.coord')
test_data = LoadData('../Data/testdata', None, data.marker_dict)

adatest = AdaboostClassification()
# clf, res, RMSE = adatest.run(data, test_data)

clf, res, RMSE = adatest.run_ble(data, test_data, True, True, True)

plt.hist(RMSE, 100)
plt.show()
#plt.savefig("miao.png")

