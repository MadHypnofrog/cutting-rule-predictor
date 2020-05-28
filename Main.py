from HTMLBuilder import HTMLBuilder
import os
from StatsCalculator import StatsCalculator
from MetaFeatureExtractor import MetaFeatureExtractor
from KNNModel import KNNModel
from MetaModel import MetaModel
import numpy as np
from StatsCollector import run_single, run_all

html_path = "D:\\html"
table_name = "\\table.html"
single_table_name = "\\table_%s.html"
kfold_table_name = "\\table_kfold.html"
ds_path = "D:\\datasets_fixed"
log_path = "stats.pickle"
meta_path = "meta.pickle"

all_stats = StatsCalculator().read_stats(open(log_path, 'rb'))

ds_names = []
stats = []

datasets = MetaFeatureExtractor().read_meta(open(meta_path, 'rb'))

#for filename in os.listdir(ds_path):
#	ds_name = os.path.splitext(filename)[0]
#	ds_names.append(ds_name)
#	stats.append(all_stats[ds_name]['f1_micro'])

#MetaModel(filepath="D:\\gridlog.txt").knn_grid(np.array(datasets), np.array(ds_names), np.array(stats))