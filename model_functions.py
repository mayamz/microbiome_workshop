import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
from statsmodels.tsa.stattools import pacf
custom_palette = list(sns.color_palette("hls", 100))
DATA_PATH = "./data/"
meta_features = ["sample", "baboon_id", "collection_date"]


def load_data():
    metadata = pd.read_csv(f"{DATA_PATH}train_metadata.csv")
    metadata["collection_date"] = pd.to_datetime(metadata["collection_date"])
    data = pd.read_csv(f"{DATA_PATH}train_data.csv")

    species = list(data.columns)
    species.remove("sample")

    data = pd.merge(data, metadata[meta_features], on = 'sample', how = 'inner')
    data = data[meta_features + species]
    return data, metadata

