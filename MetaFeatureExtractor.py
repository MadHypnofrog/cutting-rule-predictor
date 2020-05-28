import numpy as np
import pandas as pd
from pymfe.mfe import MFE
import pickle

class MetaFeatureExtractor():

	"""
		A class that extracts meta-features from datasets.

		Parameters
		----------
	"""
	def __init__(self):
		pass

	"""
		Extracts the meta-feature vector representation from a given dataset and its class labels.

		Parameters
		----------
		X : array-like, shape (n, m) or pd.DataFrame
			The dataset.
		y : array-like, shape (n, ) or pd.Series
			The class labels

		Returns
		-------
		numpy array : a vector representation of the dataset
	"""
	def transform(self, X, y):
		if isinstance(X, pd.DataFrame):
			X = X.to_numpy(dtype='int8')
		if isinstance(y, pd.Series):
			y = y.to_numpy(dtype='int32')
		mfe = MFE(groups=["general"], summary=['kurtosis', 'min', 'max', 'median', 'skewness'])
		mfe.fit(X, y)
		ft = mfe.extract()[1]
		return np.nan_to_num(np.array(ft), 0)

	"""
		Reads all the meta representations from a file.

		Parameters
		----------
		handle : python file handle
			A handle for the file with the representations.

		Returns
		-------
		array-like, shape (n_datasets, n_features) - a meta dataset
	"""
	def read_meta(self, handle):
		meta = pickle.load(handle)
		handle.close()
		return meta

	"""
		Writes all the meta representations to a file. 

		Parameters
		----------
		handle : python file handle
			A handle for the file to write the meta representations to.
		meta : array-like, shape (n_datasets, n_features)
			Meta dataset.

		Returns
		-------
		None
	"""
	def write_stats(self, handle, meta):
		pickle.dump(meta, handle)
		handle.close()
