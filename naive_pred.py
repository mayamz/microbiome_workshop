import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy
from sklearn.manifold import MDS
from scipy.spatial.distance import cdist, pdist, squareform



def shift_i_samples(data, i=1):
    """ data should contain baboon_id and collection_date """
    X_cols = [col for col in data.columns]
    y_cols = [f"y_{col}" for col in X_cols]

    data = data.sort_values(["baboon_id", "collection_date"]).reset_index(drop=True)
    data[y_cols] = data.shift(-i)
    data = data[data["baboon_id"]==data["y_baboon_id"]]

    X = data[X_cols]
    y = data[y_cols]
    y.columns = X_cols

    return X, y


def shift_i_samples_remove_gaps(data, i=1, max_gap = np.inf):
    """ data should contain baboon_id and collection_date """
    X_cols = [col for col in data.columns]
    y_cols = [f"y_{col}" for col in X_cols]

    data = data.sort_values(["baboon_id", "collection_date"]).reset_index(drop=True)
    data[y_cols] = data.shift(-i)
    data = data[data["baboon_id"] == data["y_baboon_id"]]
    data = data[(data["y_collection_date"] - data["collection_date"]).dt.days < max_gap]

    X = data[X_cols]
    y = data[y_cols]
    y.columns = X_cols

    return X, y

def calc_distance_matrix(X, y, max_gap=np.inf):
    d_matrix = cdist(X, y, metric='braycurtis').diagonal()
    pd.DataFrame(d_matrix).to_csv(f"performance{max_gap}.csv")
    return d_matrix

def plot_distances(d_matrix, max_gap):
    sns.histplot(d_matrix, bins=60, kde=True, color="steelblue")
    plt.axvline(x=np.mean(d_matrix), color='steelblue', linestyle='--', linewidth=2)

    plt.title(f"Average Distance of Model (Maximal Gap = {max_gap})\nAverage {round(np.mean(d_matrix),3)}")
    plt.xlabel("Bray-Curtis Distance")
    plt.ylabel("Number of Samples")
    plt.savefig(f"Average_Distance_gap_{max_gap}")
    plt.show()
