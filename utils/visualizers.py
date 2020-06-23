import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

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
"""
def make_pca(df):
    df_transformed = PCA(n_components=2).fit_transform(df)
    print(df_transformed)
    print(df_transformed.shape)
    plt.clf()
    plt.scatter(df_transformed[:, 0], df_transformed[:, 1])
    plt.show()