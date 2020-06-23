import numpy as np

from .knn_model import KNNModel, KNNConfig
from utils.kernels import kernels
from utils.html_builder import HTMLBuilder
from sklearn.model_selection import StratifiedKFold

class MetaModel():

    """
        A wrapper class to perform grid search on the models.

        Parameters
        ----------
        filepath : str, optional
            Path to the log file. If present, all output (configurations and losses) will be logged into that file.
    """
    def __init__(self, filepath=None):
        if filepath is not None:
            self.handle = open(filepath, 'w')
        else:
            self.handle = None

    """
        Performs a k-fold cross-validation with a specified model and writes the result to a HTMLBuilder.

        Parameters
        ----------
        html_path : str
            Path to the folder where all the html documents would be created.
        kfold_table_name : str
            Table name to use. The builder writes its content to html_path + kfold_table_name.
        model : meta-model
            A model to validate through cross-validation. Currently only KNNModel is supported.
        datasets : array-like, shape (n_datasets, n_meta_features)
            Meta-feature representation of the datasets.
        ds_names : array-like, shape(n_datasets)
            Names of the datasets.
        stats : array-like, shape (n_datasets, 2)
            Cutting rule statistics for the datasets. First column consists of float values and represents baselines, 
            the second column consists of lists and represents metric values for each cutting rule.
        modes : array-like, shape (n_modes)
            Array of prediction modes to use in format (top, radius).
        splits : int, optional
            Amount of splits to use in StratifiedKFold.
        random_states : array, optional
            Random states to use in StratifiedKFold.

        Returns
        -------
        None
    """
    def perform_kfold(self, html_path, kfold_table_name, model, datasets, ds_names, stats, modes, splits=5, random_states = [42]):
        kf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

        baselines = stats[:, 0]
        metrics = stats[:, 1]

        max_features = np.vectorize(lambda l: l.shape[0])(metrics)
        builder = HTMLBuilder(html_path, kfold_table_name, kfold=True)
        rankings = np.array(list(map(lambda metric: np.argsort(metric)[::-1], metrics)))
        total_losses = []
        for state in random_states:
            kf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=state)
            for train_index, test_index in kf.split(datasets, np.ones(datasets.shape[0])):
                datasets_train, datasets_test = datasets[train_index], datasets[test_index]
                rankings_train, rankings_test = rankings[train_index], rankings[test_index]
                metrics_test, baselines_test = metrics[test_index], baselines[test_index]
                ds_names_test, max_features_test = ds_names[test_index], max_features[test_index]
                model.fit(datasets_train, rankings_train)
                predictions = model.predict(datasets_test, max_features_test, extended=True)

                losses = []
                predictions_modes = []
                for top, radius in modes:
                    pred_mode = model.prediction_mode(predictions, top, radius)
                    loss = model.loss(metrics_test, pred_mode).reshape(-1, 1)
                    losses.append(loss)
                    predictions_modes.append(pred_mode)

                losses = np.concatenate(losses, axis=1)
                total_losses.append(losses)
                predictions_modes = np.array(list(zip(*predictions_modes)))
                builder.write_kfold(ds_names_test, metrics_test, baselines_test, predictions_modes, losses)
        builder.write_loss(np.concatenate(total_losses).T, print_categories=True, n_folds = len(random_states))

    """
        Performs a k-fold cross-validation with a specified model and returns the losses for each dataset.

        Parameters
        ----------
        model : meta-model
            A model to validate through cross-validation. Currently only KNNModel is supported.
        datasets : array-like, shape (n_datasets, n_meta_features)
            Meta-feature representation of the datasets.
        ds_names : array-like, shape(n_datasets)
            Names of the datasets.
        stats : array-like, shape (n_datasets, 2)
            Cutting rule statistics for the datasets. First column consists of float values and represents baselines, 
            the second column consists of lists and represents metric values for each cutting rule.
        modes : array-like, shape (n_modes)
            Array of prediction modes to use in format (top, radius).
        splits : int, optional
            Amount of splits to use in StratifiedKFold.

        Returns
        -------
        array-like, shape (n_modes, n_datasets) : losses for each mode and each dataset
    """
    def perform_kfold_loss_only(self, model, datasets, ds_names, stats, modes, splits=5):
        kf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

        baselines = stats[:, 0]
        metrics = stats[:, 1]

        max_features = np.vectorize(lambda l: l.shape[0])(metrics)
        rankings = np.array(list(map(lambda metric: np.argsort(metric)[::-1], metrics)))
        total_losses = []
        for train_index, test_index in kf.split(datasets, np.ones(datasets.shape[0])):
            datasets_train, datasets_test = datasets[train_index], datasets[test_index]
            rankings_train, rankings_test = rankings[train_index], rankings[test_index]
            metrics_test, baselines_test = metrics[test_index], baselines[test_index]
            ds_names_test, max_features_test = ds_names[test_index], max_features[test_index]
            model.fit(datasets_train, rankings_train)
            predictions = model.predict(datasets_test, max_features_test, extended=True)

            losses = []
            for top, radius in modes:
                pred_mode = model.prediction_mode(predictions, top, radius)
                loss = model.loss(metrics_test, pred_mode).reshape(-1, 1)
                losses.append(loss)
            losses = np.concatenate(losses, axis=1)
            total_losses.append(losses)
        return np.concatenate(total_losses).T

    """
        Runs a grid search over the KNN model space to determine the best model.

        Parameters
        ----------
        datasets : array-like, shape (n_datasets, n_meta_features)
            Meta-feature representation of the datasets.
        ds_names : array-like, shape(n_datasets)
            Names of the datasets.
        stats : array-like, shape (n_datasets, 2)
            Cutting rule statistics for the datasets. First column consists of float values and represents baselines, 
            the second column consists of lists and represents metric values for each cutting rule.
        k_range : array-like, optional
            Range of k to use in grid search.
        exp_range : array-like, optional
            Range of ranking_exp to use in grid search.
        kernels : array-like, optional
            Kernels to use in grid search.
        window_range : array-like, optional
            Windows to use in grid search.
        n_splits : int, optional
            Amount of splits to use in StratifiedKFold.
        modes : array-like, shape (n_modes), optional
            Array of prediction modes to use in format (top, radius).

        Returns
        -------
        array_like, shape (n_configs) : array of pairs (KNNConfig, array of mean of losses for each mode). 
        If self.handle is not None, those results are also written to the log file.
    """
    def knn_grid(self, datasets, ds_names, stats, k_range=np.arange(2, 20, 1), exp_range=(np.arange(0.5, 1.0, 0.05)), 
            kernels=kernels, window_range=np.arange(5e3, 1e5, 5e3), splits=5, modes=[(1, 0), (3, 0), (3, 2)]):
        datasets = np.array(datasets)
        ds_names = np.array(ds_names)
        stats = np.array(stats)
        configs = []
        for k in k_range:
            for ranking_exp in exp_range:
                model = KNNModel(k=k, mode='knn', ranking_exp=ranking_exp)
                config = model.get_config()
                losses = self.perform_kfold_loss_only(model, datasets, ds_names, stats, modes, splits=splits)
                configs.append((config, [round(np.mean(loss), 4) for loss in losses]))
                if self.handle is not None:
                    self.handle.write(('KNN: %f neighbors, ranking_exp=%f, losses: %s\n') % 
                        (k, ranking_exp, ', '.join([str(round(np.mean(loss), 4)) for loss in losses])))

        for ranking_exp in exp_range:
            for kernel in kernels:
                for window in window_range:
                    model = KNNModel(mode='parzen-fixed', window=window, kernel=kernel, ranking_exp=ranking_exp)
                    config = model.get_config()
                    losses = self.perform_kfold_loss_only(model, datasets, ds_names, stats, modes, splits=splits)
                    configs.append((config, [round(np.mean(loss), 4) for loss in losses]))
                    if self.handle is not None:
                        self.handle.write(('parzen-fixed: %f window, kernel %s, ranking_exp=%f, losses: %s\n') % 
                            (window, kernel, ranking_exp, ', '.join([str(round(np.mean(loss), 4)) for loss in losses])))

        for k in k_range:
            for ranking_exp in exp_range:
                for kernel in kernels:
                    model = KNNModel(mode='parzen-variable', k=k, kernel=kernel, ranking_exp=ranking_exp)
                    config = model.get_config()
                    losses = self.perform_kfold_loss_only(model, datasets, ds_names, stats, modes, splits=splits)
                    configs.append((config, [round(np.mean(loss), 4) for loss in losses]))
                    if self.handle is not None:
                        self.handle.write(('parzen-variable: %f neighbors, kernel %s, ranking_exp=%f, losses: %s\n') % 
                            (k, kernel, ranking_exp, ', '.join([str(round(np.mean(loss), 4)) for loss in losses])))

        return configs
