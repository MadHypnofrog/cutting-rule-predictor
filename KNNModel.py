import numpy as np
import pandas as pd
from Kernels import kernels
from scipy.spatial.distance import cdist
from sklearn.utils import shuffle

from MetaFeatureExtractor import MetaFeatureExtractor

class KNNConfig():

	"""
		A template class that is used to save configs. Useful in grid search.
		Parameters
		----------
		k : int
			k value.
		mode : str
			mode value.
		metric : str or callable
			metric value.
		kernel : str or callable
			kernel value.
		window : float
			window value.
		ranking_exp : float
			ranking_exp value.
	"""
	def __init__(self, k, mode, metric, kernel, window, ranking_exp):
		self.k = k
		self.mode = mode
		self.metric = metric
		self.kernel = kernel
		self.window = window
		self.ranking_exp = ranking_exp

class KNNModel():


	"""
		A k-nearest-neighbors based model that predicts the cutting rule ranking for a given dataset.

		Parameters
		----------
		k : int, optional
			Number of nearest neighbors to use if mode='knn' or mode='parzen-variable'.
		mode : str, optional
			Type of algorithm to use when assigning weights to the training set. Possible values are 'knn', 'parzen-fixed'
			and 'parzen-variable'.
		metric : str or callable, optional
			Distance metric to use in pairwise distances computing. If str, should be a valid metric as specified in scipy's cdist.
		kernel : str or callable, optional
			Kernel function to use when assigning weights to the training set. If str, should be a kernel specified in Kernels.
		window : float, optional
			Window width to use in parzen-fixed mode.
		ranking_exp : float, optional
			A float in range (0, 1) to use when assigning weights to rules in the ranking. Referred as x in the paper.
	"""
	def __init__(self, k=None, mode='knn', metric='euclidean', kernel=None, window=None, ranking_exp=0.95):
		if mode not in ['knn', 'parzen-fixed', 'parzen-variable']:
			raise KeyError('mode should be knn, parzen-fixed or parzen-variable, %s passed' % mode)
		if mode == 'knn':
			self.weighting = self.knn
			if k is None:
				raise KeyError('k cannot be None if mode=\'knn\' is used')
		elif mode == 'parzen-fixed':
			self.weighting = self.parzen_fixed
			if kernel is None:
				raise KeyError('kernel cannot be None if mode=\'parzen-fixed\' is used')
			if window is None:
				raise KeyError('window cannot be None if mode=\'parzen-fixed\' is used')
		else:
			self.weighting = self.parzen_variable
			if kernel is None:
				raise KeyError('kernel cannot be None if mode=\'parzen-variable\' is used')
			if k is None:
				raise KeyError('k cannot be None if mode=\'parzen-variable\' is used')
		self.k = k
		if isinstance(kernel, str):
			self.kernel = kernels[kernel]
		else:
			self.kernel = kernel
		self.metric = metric
		self.window = window
		self.ranking_exp = ranking_exp
		self.mf = MetaFeatureExtractor()

	"""
		Assigns weights to the training set based on a k-nearest-neighbors algorithm.

		Parameters
		----------
		distances : array-like, shape (n_samples)
			Distances between the input object and each object in the training set.

		Returns
		-------
		array-like, shape (n_samples) : weights for each of the objects in the training set
	"""
	def knn(self, distances):
		sorted_indices = np.argsort(distances)
		res = np.zeros(distances.shape[0])
		res[sorted_indices[:self.k]] = 1
		return res

	"""
		Assigns weights to the training set based on a k-nearest-neighbors algorithm with a Parzen-Rozenblatt window
		of fixed length.

		Parameters
		----------
		distances : array-like, shape (n_samples)
			Distances between the input object and each object in the training set.

		Returns
		-------
		array-like, shape (n_samples) : weights for each of the objects in the training set
	"""
	def parzen_fixed(self, distances):
		return np.vectorize(lambda d: self.kernel(d / self.window))(distances)
	
	"""
		Assigns weights to the training set based on a k-nearest-neighbors algorithm with a Parzen-Rozenblatt window
		of variable length.

		Parameters
		----------
		distances : array-like, shape (n_samples)
			Distances between the input object and each object in the training set.

		Returns
		-------
		array-like, shape (n_samples) : weights for each of the objects in the training set
	"""
	def parzen_variable(self, distances):
		sorted_indices = np.argsort(distances)
		w = distances[sorted_indices[self.k]]
		return np.vectorize(lambda d: self.kernel(d / w))(distances)

	"""
		Assigns weights to the each of the cutting rules in the ranking.

		Parameters
		----------
		l : array-like, shape (n_rules)
			The ranking of cutting rules. It is assumed that the rules are already sorted in descending order according
			to some metric.
		weight : float
			Weight of the object assigned by the model.

		Returns
		-------
		dict of pairs (rule_n, weight) : weights for each of the rules
	"""
	def weight_ranking(self, l, weight):
		d = {}
		for i in range(l.shape[0]):
			d[l[i]] = (self.ranking_exp ** i) * weight
		return d

	"""
		Merges the rankings from all of the objects in the training set into a prediction ranking.

		Parameters
		----------
		weights : array-like, shape (n_objects)
			Weights for each of the objects in the training set.

		Returns
		-------
		array of pairs (rule_n, weight) sorted by weight : the predicted ranking
	"""
	def merge(self, weights):
		result = {}
		for i in range(weights.shape[0]):
			d = self.weight_ranking(self.lists[i], weights[i])
			for key in d:
				result[key] = result.get(key, 0) + d[key]
		return np.array([[k, v] for k, v in sorted(result.items(), key=lambda item: item[1])][::-1])

	"""
		Fits the model. For big training sets consider passing the meta-feature representation of all datasets to avoid
		using a lot of memory. Note that this method would NOT calculate the cutting rule rankings and they should be 
		passed explicitly as y.

		Parameters
		----------
		X : array-like, shape (n_datasets) or (n_datasets, n_meta_features)
			Datasets in the training set or their meta-feature representation.
		y : array-like, shape (n_datasets, )
			Cutting rule rankings for datasets in the training set.

		Returns
		-------
		None
	"""
	def fit(self, X, y):
		if len(X.shape) == 1:
			self.X = np.vectorize(lambda x: self.mf.transform(x))(X)
		else:
			self.X = X
		self.lists = y

	"""
		Predicts the cutting rule rankings for given datasets. For big test sets consider passing the meta-feature
		representation of all datasets to avoid using a lot of memory. 

		Parameters
		----------
		X : array-like, shape (n_datasets) or (n_datasets, n_meta_features)
			Datasets in the test set or their meta-feature representation.
		max_features : array-like, shape (n_datasets, )
			The maximum amount of features to use in cutting rules for each dataset. Usually the amount of features
			in the dataset goes here, but something smaller can be set as well.
		extended : bool, optional
			If False, only one integer x is returned that represents the best cutting rule with x+1 features. If True,
			the whole ranking would be returned.

		Returns
		-------
		array-like, shape (n_datasets, ) : predictions for the datasets in the test set
	"""
	def predict(self, X, max_features, extended=False):
		result = []
		if len(X.shape) == 1:
			X_tr = np.vectorize(lambda x: self.mf.transform(x))(X)
		else:
			X_tr = X
		for obj, max_feature in zip(X_tr, max_features):
			distances = cdist(self.X, [obj], metric=self.metric).reshape(-1)
			ranking = self.merge(self.weighting(distances))
			ranking = ranking[ranking[:, 0] < max_feature]
			if extended:
				result.append(ranking)
			else:
				result.append(int(ranking[0][0]))
		return np.array(result)

	"""
		Extracts a prediction in a specified mode from the extended ranking.

		Parameters
		----------
		predictions : array-like, shape (n_datasets, )
			Predictions for the datasets in the test set.
		top : int
			Amount of best cutting rules to return.
		radius : int
			Amount of cutting rules adjacent to the best ones that would be returned. For example, if radius=3 and the
			best cutting rule is 55, [52, 53, 54, 55, 56, 57, 58] would be returned for that rule.

		Returns
		-------
		array-like, shape (n_datasets, ) : predictions for the datasets in the test set with the mode applied
	"""
	def prediction_mode(self, predictions, top, radius, random=False):
		result = []
		for i in range(predictions.shape[0]):
			ranking = predictions[i][:, 0].astype(int)
			max_f = np.max(ranking) + 1
			if random:
				ranking = shuffle(ranking)
			indices = ranking[:top]
			result.append(np.concatenate([[x for x in range(idx - radius, idx + radius + 1, 1) if x >= 0 and x < max_f] for idx in indices]))
		return np.array(result)

	"""
		Calculates the loss function as described in the paper.

		Parameters
		----------
		stats : array-like, shape (n_datasets, )
			Statistics for the datasets in the test set. Should be unsorted unlike the rankings.
		predictions : array-like, shape (n_datasets, )
			Predictions for the datasets in the test set.

		Returns
		-------
		array-like, shape (n_datasets) : values of loss function for each dataset in the test set
	"""
	def loss(self, stats, predictions):
		result = []
		for i in range(stats.shape[0]):
			max_metric = np.max(stats[i])
			min_metric = np.min(stats[i])
			pred_values = stats[i][predictions[i]]
			max_pred = predictions[i][np.argsort(pred_values)[-1]]
			pred = stats[i][max_pred]
			if max_metric - min_metric == 0:
				result.append(0)
			else:
				result.append(1 - (pred - min_metric) / (max_metric - min_metric))
		return np.array(result)