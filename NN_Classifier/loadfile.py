__author__ = 'Xiaolong Shen'
__version__ = "0.1.0"


import numpy as np


class DataStorage:
	def __init__(self, wp_filepath, wifi_filepath, wifi_list, wp_pos, wp_ind, wifi_matrix):
		self.wp_filepath = wp_filepath
		self.wifi_filepath = wifi_filepath
		self.wifi_list = wifi_list
		self.wp_pos = wp_pos
		self.wp_ind = wp_ind
		self.wifi_matrix = wifi_matrix


class LoadData:
	"""
		Generate Necessary Data For WiFI And Wp Files Extraction, including:
			self.wp_filepath : filepath of wp
			self.wifi_filepath : the same
			self.wifi_list : wifi mac address contained in the files
			self.wp_pos : position of each way point
			self.wp_ind : index of waypoint matching the wifi
			self.wifi_matrix : matrix-lize wifi data in order of wifi_list
	"""
	def __init__(self):
		self.wp_filepath = ''
		self.wifi_filepath = ''
		self.wifi_list = []

	@classmethod
	def test_print(cls):
		print "Successfully loaded"

	# @classmethod
	def extract(self, wp_filepath, wifi_filepath):
		self._extract(wp_filepath, wifi_filepath)
		data = DataStorage(wp_filepath, wifi_filepath, self.wifi_list, self.wp_pos, self.wp_ind, self.wifi_matrix)
		return data

	def _extract(self, wp_filepath, wifi_filepath):
		wp_pos, wp_ind = self._extract_wp(wp_filepath)
		self.wp_pos = wp_pos
		self.wp_ind = wp_ind
		self.wifi_matrix = self._extract_wifi_with_wp(wifi_filepath)

	def _extract_wp(self, wp_filepath):
		"""
		:rtype : waypoint position, waypoint index
		:param wp_filepath: 
		"""
		f = open(wp_filepath)
		tmpwppos = np.zeros((80000, 2))
		pospointer = 0
		wpind = [0]
		for l in f:
			if len(l) == 0:
				continue
			ls = l.split()
			if len(ls) == 2:
				tmpwppos[pospointer, 0] = float(ls[0])
				tmpwppos[pospointer, 1] = float(ls[1])
				pospointer += 1
			elif len(ls) == 1:
				wpind.append(int(ls[0]))
		# wp_pos = zeros((pospointer, 2))
		wp_pos = tmpwppos[0:pospointer, :]
		wp_ind = np.array(wpind)
		f.close()
		del tmpwppos
		return wp_pos, wp_ind

	def _extract_wifi_with_wp(self, wifi_filepath):
		"""
		:param : wifi_filepath
		:rtype : Matrix of Wifi In WifiList Order
		"""
		self.wifi_list = self._buildwifilist(wifi_filepath)
		f = open(wifi_filepath)
		wifimatrix = np.zeros((len(self.wp_ind), len(self.wifi_list)))
		indpointer = 0
		for l in f:
			ls = l.split()
			if indpointer < len(self.wp_ind) and int(ls[0]) == self.wp_ind[indpointer]:
				if len(ls) < 3:
					indpointer += 1
					continue
				for n in range(1, len(ls), 2):
					mac = ls[n]
					ss = int(ls[n + 1])
					insertpos = self.wifi_list.index(mac)
					try:
						wifimatrix[indpointer, insertpos] = ss
					except Exception,e:
						print e
				indpointer += 1
		return wifimatrix

	def _buildwifilist(self, filepath):
		"""
		:param : File Path of Wifi File
		:rtype : List of Wifi Mac Address
		"""
		wifilist = []
		f = open(filepath)
		for l in f:
			ls = l.split()
			if len(ls) < 3:
				continue;
			for i in ls[1::2]:
				if i not in wifilist:
					wifilist.append(i)
		f.close()
		return wifilist

