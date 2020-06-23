import pandas as pd
import numpy as np
import os

from utils.html_builder import HTMLBuilder
from .stats_calculator import StatsCalculator

calc = StatsCalculator()

"""
    A class that provides utility functions used to collect the statistics.
"""

"""
    Collects the statistics on a single dataset and writes them to a HTMLBuilder.

    Parameters
    ----------
    ds_path : str
        Path to the folder with the datasets.
    ds_name : str
        The dataset name.
    builder : HTMLBuilder
        HTMLBuilder to write the results to.
    metrics : array-like
        Array of metrics to use in calculations. All elements should be valid metrics to use in 
        StatsCalculator.stats_for_rules.
    stats_d : dict of pairs (metric_name, [baseline, metrics]), optional
        Statistics for the dataset (same format as in StatsCalculator.read_stats or write_stats). If present, no 
        calculation is done and the statistics are just written to the HTMLBuilder.

    Returns
    -------
    dict of pairs (metric_name, [baseline, metrics]) - statistics for the dataset. Returns stats_d if it is not None.
"""
def process_ds(ds_path, ds_name, builder, metrics, stats_d=None):
    df = pd.read_csv(ds_path + "\\" + ds_name + ".csv", dtype='int8')
    data, target = df.drop('class', axis=1), df['class']
    if stats_d is None:
        stats_d = {}
        stats = calc.stats_for_rules(data, target, metrics)
        baselines = calc.baseline(data, target, metrics)
        for i in range(len(metrics)):
            stats_d[metrics[i]] = [baselines[i], stats[i]]
    builder.write_ds(df, ds_name, stats_d)
    return stats_d

"""
    Iterates through all datasets in a specified folder, collects their statistics and writes them to a HTMLBuilder.

    Parameters
    ----------
    html_path : str
        Path to the folder where all the html documents would be created.
    table_name : str
        Table name to use. The builder writes its content to html_path + table_name.
    ds_path : str
        Path to the folder with the datasets.
    start : str, optional
        If not None, all datasets that are lexicographically smaller than start would be skipped.
    log : str, optional
        Path to the log file. If present, calculated statistics would be saved into that file.
    metrics : array-like, optional
        Array of metrics to use in calculations. All elements should be valid metrics to use in 
        StatsCalculator.stats_for_rules.
    stats : dict of pairs (ds_name, dict of pairs (metric_name, [baseline, metrics])), optional
        Precomputed stats for the datasets. Would be also saved to the log file if it is present. See StatsCalculator 
        for dict format.

    Returns
    -------
    None
"""
def run_all(html_path, table_name, ds_path, start=None, log=None, metrics=['f1_micro'], stats=None):
    if start is None:
        builder = HTMLBuilder(html_path, table_name)
    else:
        builder = HTMLBuilder(html_path, table_name, new=False)
    if log is not None:
        logh = open(log, 'wb')
    skip = True
    stats_write = {}
    for filename in os.listdir(ds_path):
        ds_name = os.path.splitext(filename)[0]
        if start != None and ds_name != start and skip:
            continue
        skip = False
        if stats is not None and ds_name in stats:
            process_ds(ds_path, ds_name, builder, metrics, stats_d=stats[ds_name])
        else:
            stats_write[ds_name] = process_ds(ds_path, ds_name, builder, metrics)
    builder.write_ending()
    if stats_write != {}:
        if stats is not None:
            stats_write.update(stats)
        calc.write_stats(logh, stats_write)

"""
    Collects statistics of a single dataset and writes them into a separate table.

    Parameters
    ----------
    html_path : str
        Path to the folder where all the html documents would be created.
    single_table_name : str
        Table name to use. The builder writes its content to html_path + (single_table_name % ds_name).
    ds_path : str
        Path to the folder with the datasets.
    ds_name : str
        The dataset name.
    metrics : array-like, optional
        Array of metrics to use in calculations. All elements should be valid metrics to use in 
        StatsCalculator.stats_for_rules.
    stats_d : dict of pairs (metric_name, [baseline, metrics]), optional
        Statistics for the dataset (same format as in StatsCalculator.read_stats or write_stats). If present, no 
        calculation is done and the statistics are just written to the HTMLBuilder.

    Returns
    -------
    None
"""
def run_single(html_path, single_table_name, ds_path, ds_name, metrics=['f1_micro'], stats_d=None):
    builder = HTMLBuilder(html_path, single_table_name % ds_name)
    process_ds(ds_path, ds_name, builder, metrics, stats_d=stats_d)
    builder.write_ending()
