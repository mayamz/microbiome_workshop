import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.manifold import MDS
from scipy.spatial.distance import braycurtis

custom_palette = list(sns.color_palette("hls", 100))
DATA_PATH = "./data/"


def load_data():
    metadata = pd.read_csv(f"{DATA_PATH}train_metadata.csv")
    data = pd.read_csv(f"{DATA_PATH}train_data.csv")
    return data, metadata

# Visulize metadata
def visualize_metadata(metadata):
    # samples_per_baboon(metadata)
    time_diff_samples(metadata)


def samples_per_baboon(metadata):
    # samples per baboon df
    sample_per_baboon = metadata.groupby("baboon_id")[["sample"]].count()
    sample_per_baboon.index = sample_per_baboon.index.str.replace("Baboon_", "")
    sample_per_baboon = sample_per_baboon.sort_values(by = "sample", ascending = False)

    # Hist
    sns.histplot(sample_per_baboon, bins = 60)
    plt.title("Sample per Baboon")
    plt.xlabel("Number of Samples per Baboon")
    plt.ylabel("Number of Baboons")
    plt.show()

    # sample per baboon scatter plot
    fig = plt.figure(figsize = (15, 10))
    plt.scatter(sample_per_baboon.index, sample_per_baboon["sample"])
    plt.xticks(rotation = 45, size = 7)
    plt.title("Number of Samples per Baboon ID")
    plt.xlabel("Baboon ID")
    plt.ylabel("Number of Samples")
    plt.show()


def time_diff_samples(metadata):
    # plot diff per baboon
    metadata["baboon_id"] = metadata["baboon_id"].str.replace("Baboon_","").astype(int)
    metadata = metadata.sort_values(by="baboon_id")
    fig = plt.figure(figsize = (15, 10))
    metadata["collection_date"] = pd.to_datetime(metadata["collection_date"])
    for i, baboon in enumerate(metadata["baboon_id"].unique()):
        baboon_samples = metadata[metadata["baboon_id"] == baboon].sort_values(by = "collection_date")
        baboon_samples['difference'] = baboon_samples["collection_date"].diff().dt.days
        sns.boxplot(x = i, y = baboon_samples['difference'], color = custom_palette[i])
    plt.title("Time Difference Between Sequential Samples")
    fig.canvas.draw()

    plt.xticks(range(len(metadata["baboon_id"].unique())), metadata["baboon_id"].unique(), rotation = 'vertical')
    plt.xlabel("Baboon")
    plt.ylabel("Days Between Sequential Samples")
    plt.show()

    # plot total diff per baboon
    fig = plt.figure(figsize = (15, 10))
    metadata["collection_date"] = pd.to_datetime(metadata["collection_date"])
    diffs = pd.Series()
    for baboon in metadata["baboon_id"].unique():
        baboon_samples = metadata[metadata["baboon_id"] == baboon].sort_values(by = "collection_date")
        baboon_samples['difference'] = baboon_samples["collection_date"].diff().dt.days
        diffs = pd.concat([diffs, baboon_samples['difference']])

    sns.histplot(diffs)
    plt.title("Time Difference Between Sequential Samples")
    plt.xlabel("Days Between Sequential Samples")
    plt.ylabel("Samples")
    plt.show()

    sns.histplot(diffs[diffs > 200])
    plt.show()


def FFQ_plot(metadata):
    sns.scatterplot(x=metadata["diet_PC1"], y=metadata["diet_PC2"], palette = custom_palette[::10], hue =metadata["social_group"])
    plt.title("By group")
    plt.show()

    sns.scatterplot(x = metadata["diet_PC1"], y = metadata["diet_PC2"], palette = custom_palette[::10],
                    hue = metadata["season"])
    plt.title("By season")
    plt.show()

    sns.scatterplot(x = metadata["diet_PC1"], y = metadata["diet_PC2"], palette = custom_palette,
                    hue = metadata["rain_month_mm"])
    plt.title("By rain")
    plt.show()

    sns.scatterplot(x = metadata["diet_PC1"], y = metadata["diet_PC2"], palette = custom_palette[0:90:5],
                    hue = metadata["month"])
    plt.title("By month")
    plt.show()

# visualize data
def visualize_data(data):
    distance_matrix(data)

def distance_matrix(data):
    distance_matrix = np.zeros((len(data), len(data)))
    for index1, sample1 in data.iterrows():
        for index2, sample2 in data.iterrows():
            distance_matrix[index1, index2] = braycurtis(data.iloc[index1, 1:], data.iloc[index2, 1:])
    print(distance_matrix)


def main():
    data, metadata = load_data()
    visualize_metadata(metadata)
    visualize_data(data)
    FFQ_plot(metadata)

if __name__ == '__main__':
    main()
