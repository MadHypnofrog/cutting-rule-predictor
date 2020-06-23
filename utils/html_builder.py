import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from datasets.stats_calculator import StatsCalculator

#TODO: customize headers to use with multiple metrics
header = ''.join(['<html><head><link rel=\"stylesheet\" href=\"styles.css\"></head>',
    '<body><table><tr><th>Dataset name</th><th>Baseline (micro)</th><th>Graph (micro)</th>',
    '<th>Best F1 (micro)</th><th>Number of peaks/valleys (micro)</th>',
    '</tr>'])
header_kfold = ''.join(['<html><head><link rel=\"stylesheet\" href=\"styles.css\"></head>',
    '<body><table><tr><th>Dataset name</th><th>Graph</th><th>Min value / max value</th>',
    '<th>Loss1</th><th>Loss2</th><th>Loss3</th></tr>'])
tr = '<tr>%s</tr>'
td = '<td>%s</td>'
img = '<img class=\"graph\" src=\"images\\%s\">'

class HTMLBuilder():

    """
        An utility class that builds HTML tables with various statistics.

        Parameters
        ----------
        path : str
            Path to the root folder that would be used.
        filename : str
            Filename to write the table to.
        new : bool, optional
            If True, a new file is created. If False, the content is appended to whatever exists in the file.
        kfold : bool, optional
            If True, generates a k-fold table. If False, generates a table with graphs and peaks/valleys.
    """
    def __init__(self, path, filename, new=True, kfold=False):
        self.path = path
        if new:
            self.handle = open(path + filename, 'w')
            if kfold:
                self.handle.write(header_kfold)
            else:
                self.handle.write(header)
        else:
            self.handle = open(path + filename, 'a')
        self.calc = StatsCalculator()

    """
        Builds the graph for a dataset.

        Parameters
        ----------
        ds_name : str
            The dataset name.
        metrics : array-like
            Metrics of some model performance depending on the amount of features selected by the filter.
            metrics[i] is supposed to be the metric with i+1 features selected.
        baseline : float
            Baseline metric without feature selection (i. e. on the original dataset).
        metric_name : str
            Name of the metric used.
        dot : array-like, optional
            If present, marks the indices specified in dot on the graph (used only in k-fold).

        Returns
        -------
        None, creates a png file at self.path + "\\images\\" + ds_name + "_" + metric_name + ".png" if dot
        is None, and at self.path + "\\images\\kfold\\" + ds_name + "_" + metric_name + "_kfold.png" if dot
        is present.
    """
    def build_graph(self, ds_name, metrics, baseline, metric_name, dot=None):
        plt.clf()
        plt.xlabel('Число признаков')
        plt.ylabel('Метрика')
        numbers = [i + 1 for i in range(len(metrics))]
        plt.plot(numbers, metrics)
        plt.hlines([baseline], 1, len(metrics))
        filename = self.path + "\\images\\" + ds_name + "_" + metric_name + ".png"
        if dot is not None:
            plt.plot(dot + 1, metrics[dot], 'ro')
            filename = self.path + "\\images\\kfold\\" + ds_name + "_" + metric_name + "_kfold.png"
        plt.savefig(filename)

    """
        Builds the histogram based on loss values.

        Parameters
        ----------
        losses : array-like, shape (n_losses, n_datasets)
            Loss values for all of the datasets. Each row is a single loss of each of the datasets.

        Returns
        -------
        None, creates a png file at self.path + "\\images\\kfold\\hist.png" with the histogram.
    """
    def build_hist(self, losses):
        plt.clf()
        matplotlib.rcParams.update({'font.size': 18})
        num_losses = len(losses)
        fig, axs = plt.subplots(num_losses, 1, figsize=[12.8, 9.6], tight_layout=True)
        for i in range(num_losses):
            axs[i].set_xlabel('Значение функции потерь')
            axs[i].set_ylabel('Процент')
            axs[i].hist(losses[i], bins=100, weights=np.ones(len(losses[i])) / len(losses[i]))
            axs[i].get_yaxis().set_major_formatter(PercentFormatter(1))
        fig.tight_layout(pad=8.0)
        filename = self.path + "\\images\\kfold\\hist.png"
        plt.savefig(filename)

    """
        Writes html for the default mode.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset.
        ds_name: str
            The name of the dataset.
        stats : dict of pairs (metric_name, [baseline, metrics])
            Statistics for the dataset (same format as in StatsCalculator.read_stats or write_stats).

        Returns
        -------
        None
    """
    def write_ds(self, df, ds_name, stats):
        for metric_name, (baseline, metrics) in stats.items():
            self.build_graph(ds_name, metrics, baseline, metric_name)

        ds_meta = self.gen_meta(df, ds_name)
        stats = ''.join([self.gen_stats(ds_name, metrics, baseline, self.calc.peaks_and_valleys(metrics), metric_name) 
            for metric_name, (baseline, metrics) in stats.items()])
        html = tr % ''.join([ds_meta, stats])
        self.handle.write(html)

    """
        Writes html for k-fold procedure.

        Parameters
        ----------
        ds_names : array-like, shape (n_datasets)
            Dataset names.
        metrics : array-like, shape (n_datasets, )
            Array of metrics for cutting rules for all of the datasets.
        baselines : array-like, shape (n_datasets)
            Baselines for all of the datasets.
        predictions : array-like, shape (n_datasets, )
            Predictions for all of the datasets. Can have multiple predictions.
        losses : array-like, shape (n_datasets, )
            Losses for all of the datasets. If predictions has multiple predictions, losses should have the same
            amount of losses for each dataset.

        Returns
        -------
        None
    """
    def write_kfold(self, ds_names, metrics, baselines, predictions, losses):
        for i in range(ds_names.shape[0]):
            ds_name = ds_names[i]
            metric = metrics[i]
            baseline = baselines[i]
            prediction = predictions[i]
            loss = losses[i]
            self.write_single_kfold(ds_name, metric, baseline, prediction, loss)

    """
        Writes html for a single dataset in k-fold procedure.

        Parameters
        ----------
        ds_name : str
            Dataset name.
        metrics : array-like
            Metrics for cutting rules.
        baseline : float
            Baseline.
        prediction : array-like
            Predictions. Can have multiple predictions. Only the first one would be passed as dot in build_graph.
        losses : array-like
            Losses. If predictions has multiple predictions, losses should have the same amount of losses.

        Returns
        -------
        None
    """
    def write_single_kfold(self, ds_name, metrics, baseline, prediction, losses):
        self.build_graph(ds_name, metrics, baseline, 'micro', prediction[0])

        ds_meta = td % ds_name
        graph = td % (img % ''.join(['kfold\\', ds_name, '_micro_kfold.png']))
        stats = td % ''.join([str(round(np.min(metrics), 4)), '\\', str(round(np.max(metrics), 4))])
        ls = []
        for i in range(len(losses)):
            ls.append(td % ''.join(['pred value = ', str(round(np.max(metrics[prediction[i]]), 4)), 
                '\nloss = ', str(round(losses[i], 4))]))
        html = tr % ''.join([ds_meta, graph, stats] + ls)
        self.handle.write(html)

    """
        Generates a part of an HTML row with the statistics. Used with kfold = False.

        Parameters
        ----------
        ds_name : str
            Dataset name.
        metrics : array-like
            Metrics for cutting rules.
        baseline : float
            Baseline.
        pv_num : (int, int)
            Number of peaks and valleys.
        metric_name : str
            Metric name used in metrics.

        Returns
        -------
        str : HTML with the statistics
    """
    def gen_stats(self, ds_name, metrics, baseline, pv_num, metric_name):
        bs = td % str(round(baseline, 4))
        graph = td % (img % ''.join([ds_name, '_', metric_name, '.png']))
        best = td % str(round(np.max(metrics), 4))
        peaks, valleys = pv_num
        pv = td % ''.join([str(peaks), '\\', str(valleys)])
        return ''.join([bs, graph, best, pv])

    """
        Generates a part of an HTML row with the metadata. Used with kfold = False.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset.
        ds_name : str
            Dataset name.

        Returns
        -------
        str : HTML with the metadata
    """
    def gen_meta(self, df, ds_name):
        num_classes = str(df['class'].unique().shape[0])
        num_features = str(df.shape[1] - 1)
        num_objects = str(df.shape[0])
        return td % ''.join([ds_name, '<br>', num_classes, ' classes, ', num_features, ' features, ', 
            num_objects, ' objects'])

    """
        Writes statistic based on losses of all datasets. 

        Parameters
        ----------
        losses : array-like, shape (n_losses, n_datasets)
            Loss values for all of the datasets. Each row is a single loss of each of the datasets.
        print_categories : bool, optional
            If True, prints the amount of datasets in each of the categories described in the end of the paper.
        n_fold : int, optional
        	Number of k-folds used.

        Returns
        -------
        None
    """
    def write_loss(self, losses, print_categories=False, n_folds=None):
        self.build_hist(losses)
        buckets = np.split(losses, n_folds, axis=1)
        means = np.apply_along_axis(lambda z: np.mean(z), 2, buckets).T
        losses_rounded = [td % '/'.join([str(round(np.mean(m), 4)), str(round(np.var(m), 4))]) for m in means]
        if print_categories:
            print(len([l for l in losses[0] if l == 0.0]) / n_folds)
            print(len([l for l in losses[1] if l == 0.0]) / n_folds)
            print(len([l for l in losses[2] if l == 0.0]) / n_folds)
            print(len([l for l in losses[2] if l < 0.2]) / n_folds)
            print(len([l for l in losses[2] if l < 0.5]) / n_folds)
            print(len([l for l in losses[2] if l < 1]) / n_folds)

        hist = tr % ''.join([td % (img % 'kfold\\hist.png'), td % '', td % '', td % '', td % '', td % ''])
        loss = tr % ''.join([td % '', td % '', td % ''] + losses_rounded)
        self.handle.write(hist)
        self.handle.write(loss)

    """
        Writes the ending of the table.

        Parameters
        ----------
        None

        Returns
        -------
        None
    """
    def write_ending(self):
        self.handle.write("</table></body></html>")
        self.handle.close()
