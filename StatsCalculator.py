import numpy as np
import pandas as pd
from ITMO_FS.filters.univariate.measures import information_gain, select_k_best
from ITMO_FS.filters.univariate import UnivariateFilter
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


class StatsCalculator():

	"""
		A class that provides means to calculate the statistics for cutting rules.

		Parameters
		----------
		features_threshold : int, optional
			Maximum amount of features to leave in the dataset.
		splits : int, optional
			Amount of splits to use in KFold while evaluating the metric.
		random_state : int, optional
			Random state to use in KFold.
		model : object
			A machine learning model to use in metric evaluation. Should be a valid model that can be used in sklearn's cross_validate.
	"""
	def __init__(self, features_threshold=200, splits=5, random_state=42, model=LogisticRegression(max_iter=100000)):
		self.features_threshold = features_threshold
		self.kf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
		self.model = model

	"""
		Calculates metrics for all possible rules for a dataset.

		Parameters
		----------
		data : array-like or pd.DataFrame
			The dataset.
		target : array-like or pd.Series
			The class values.
		metrics : list of str
			Metrics that should be calculated. All values of this list should be valid metrics for sklearn's cross_validate.

		Returns
		-------
		numpy array, shape (n_metrics, features_threshold) - metrics for all cutting rules
	"""
	def stats_for_rules(self, data, target, metrics):
		features = data.shape[1]
		max_features = min(self.features_threshold, features)
		result = []
		for features_n in range(1, max_features + 1, 1):
			f = UnivariateFilter(information_gain, select_k_best(features_n))
			f.fit(data, target)
			result.append(self.baseline(data.iloc[:, f.selected_features], target, metrics).reshape(-1, 1))
		return np.concatenate(result, axis=1)

	"""
		Calculates machine learning model metrics for a dataset.

		Parameters
		----------
		data : array-like or pd.DataFrame
			The dataset.
		target : array-like or pd.Series
			The class values.
		metrics : list of str
			Metrics that should be calculated. All values of this list should be valid metrics for sklearn's cross_validate.

		Returns
		-------
		numpy array, shape (n_metrics) - metrics list
	"""
	def baseline(self, data, target, metrics):
		cv = cross_validate(self.model, data, target, cv=self.kf, scoring=metrics)
		return np.array([np.mean(cv['test_' + s]) for s in metrics])


	"""
		Calculates the amount of local maxima and minima in an array of metric values. This was used to check if it is possible
		to use smoothing on the function and is barely mentioned in the paper.

		Parameters
		----------
		metrics : array-like
			Metric values for cutting rules.

		Returns
		-------
		(int, int) - amount of local maxima and minima

	"""
	def peaks_and_valleys(self, metrics):
		peaks, valleys = 0, 0
		for idx in range(2, metrics.shape[0] - 1, 1):
			prv = idx - 1
			nxt = idx + 1
			if metrics[prv] == metrics[idx] and metrics[idx] == metrics[nxt]:
				continue
			if metrics[idx] >= metrics[prv] and metrics[idx] >= metrics[nxt]:
				peaks += 1
			if metrics[idx] <= metrics[prv] and metrics[idx] <= metrics[nxt]:
				valleys += 1
		return peaks, valleys

	"""
		Performs a t-SNE algorithm and visualizes the result. Was used to see whether the dataset was separable at all,
		not mentioned in the paper.

		Parameters
		----------
		df : array-like or pd.DataFrame
			The dataset.

		Returns
		-------
		none, opens a pyplot window with the graph

	"""
	def make_tsne(df):
		N = 2500

		np.random.seed(42)
		rndperm = np.random.permutation(df.shape[0])

		df_subset = df.loc[rndperm[:N],:].copy()
		classes = len(df_subset['class'].unique())
		data_subset = df_subset.drop('class', axis=1)

		tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
		tsne_results = tsne.fit_transform(data_subset)

		df_subset['tsne-2d-one'] = tsne_results[:,0]
		df_subset['tsne-2d-two'] = tsne_results[:,1]

		plt.figure(figsize=(8,5))
		sns.scatterplot(
			x="tsne-2d-one", y="tsne-2d-two",
			hue="class",
			palette=sns.color_palette("hls", classes),
			data=df_subset,
			legend="full",
			alpha=0.8
		)
		plt.show()

	"""
		Reads all the cutting rules stats from a file.

		Parameters
		----------
		handle : python file handle
			A handle for the file with the stats.

		Returns
		-------
		dict of pairs (ds_name, dict of pairs (metric_name, [baseline, metrics])) - statistics for each dataset
	"""
	def read_stats(self, handle):
		stats = pickle.load(handle)
		handle.close()
		return stats

	"""
		Writes all the stats to a file. 

		Parameters
		----------
		handle : python file handle
			A handle for the file to write the stats to.
		stats : dict of pairs (ds_name, dict of pairs (metric_name, [baseline, metrics]))
			Statistics for each dataset.

		Returns
		-------
		None
	"""
	def write_stats(self, handle, stats):
		pickle.dump(stats, handle)
		handle.close()
