from utils.html_builder import HTMLBuilder
import os
from datasets.stats_calculator import StatsCalculator
from datasets.meta_feature_extractor import MetaFeatureExtractor
from models.knn_model import KNNModel
from models.meta_model import MetaModel
import numpy as np
from datasets.stats_collector import run_single, run_all
from utils.visualizers import make_pca

html_path = "D:\\html"
table_name = "\\table.html"
single_table_name = "\\table_%s.html"
kfold_table_name = "\\table_kfold.html"
ds_path = "D:\\datasets_fixed"
log_path = "data\\stats.pickle"
meta_path = "data\\meta.pickle"

all_stats = StatsCalculator().read_stats(open(log_path, 'rb'))

ds_names = []
stats = []

datasets = MetaFeatureExtractor().read_meta(open(meta_path, 'rb'))

for filename in os.listdir(ds_path):
	ds_name = os.path.splitext(filename)[0]
	ds_names.append(ds_name)
	stats.append(all_stats[ds_name]['f1_micro'])

MetaModel().perform_kfold(html_path, kfold_table_name, KNNModel(mode='knn', k=4, ranking_exp=0.5), np.array(datasets), np.array(ds_names), np.array(stats),
	[(1, 0), (3, 0), (3, 2)], random_states=[42, 3284, 246820, 34281, 58, 121124, 90, 4848, 1, 85282254])
#make_pca(np.array(datasets))
#run_single(html_path, single_table_name, ds_path, 'lsvt', stats_d=all_stats['lsvt'])
