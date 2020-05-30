import numpy as np
import pandas as pd

class DatasetTransformer():

    """
        A class that transforms datasets into binary form (i. e. the resulting features only consist of 0 and 1's).

        Parameters
        ----------
        objects_threshold : int, optional
            The minimal around of different values a feature has to have to be considered numeric. Useful on datasets 
            with low object count.
        objects_classes : dict, optional
            A dictionary that contains pairs of (object_count, num_classes). If a dataset has more than object_count 
            objects, any numeric features in it would be discretized into num_classes bins. The more objects a dataset 
            has, the more bins it requires to maintain classification quality.
        objects_divisor : int, optional
            The threshold divisor. If a feature has more than num_objects / objects_divisor different values, it is 
            considered numeric. Gets overrided by objects_threshold so that datasets with low object count would not 
            treat every feature as numeric.
    """
    def __init__(self, objects_threshold=20, objects_classes={0: 3, 100: 5, 1000: 10, 5000: 15}, objects_divisor=100):
        self.objects_threshold = objects_threshold
        self.objects_classes = objects_classes
        self.objects_divisor = objects_divisor

    """
        Transforms the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset. Class label is expected to be labeled by 'class'.

        Returns
        -------
        pd.DataFrame : the transformed dataset in binary form
    """
    def transform(self, df):
        n_objects = df.shape[0]
        threshold = max(int(n_objects / self.objects_divisor), self.objects_threshold)
        for objects, classes in self.objects_classes:
            if n_objects > objects:
                num_classes = classes
            else:
                break
        res = pd.DataFrame()
        for col in df.columns:
            if col != 'class':
                values = df[col].unique().shape[0]
                if values == 2 and df.dtypes[col] != 'object':
                    mapping = {df[col].unique()[0]: 0, df[col].unique()[1]: 1}
                    res = pd.concat([res, df[col].map(mapping)], axis=1)
                elif (values > 2 and values <= threshold) or (values == 2 and df.dtypes[col] == 'object'):
                    ohe = pd.get_dummies(df[col], prefix=col)
                    res = pd.concat([res, ohe], axis=1)
                else:
                    ohe = pd.get_dummies(pd.qcut(df[col], q=num_classes, labels=False, precision=0, duplicates='drop'), 
                        prefix=col)
                    if ohe.shape[1] == 1:
                        ohe = pd.get_dummies(df[col], prefix=col)
                    res = pd.concat([res, ohe], axis=1)
        res.concat([res, df['class']], axis=1)
        return res
