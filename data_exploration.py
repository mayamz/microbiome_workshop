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
meta_features = ["sample", "baboon_id", "season", "collection_date"]


def load_data():
    metadata = pd.read_csv(f"{DATA_PATH}train_metadata.csv")
    metadata["collection_date"] = pd.to_datetime(metadata["collection_date"])
    data = pd.read_csv(f"{DATA_PATH}train_data.csv")

    species = list(data.columns)
    species.remove("sample")

    data = pd.merge(data, metadata[meta_features], on = 'sample', how = 'inner')
    data = data[meta_features + species]
    return data, metadata


# Visulize metadata
def visualize_metadata(metadata):
    samples_per_baboon(metadata)
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
    metadata["baboon_id"] = metadata["baboon_id"].str.replace("Baboon_", "").astype(int)
    metadata = metadata.sort_values(by = "baboon_id")
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
    plt.legend(loc='center left', bbox_to_anchor=(0.78, 0.29))
    plt.title("By group")
    plt.savefig("By group")
    plt.show()

    sns.scatterplot(x = metadata["diet_PC1"], y = metadata["diet_PC2"], palette = custom_palette[::10],
                    hue = metadata["season"])
    plt.title("By season")
    plt.savefig("By season")
    plt.show()

    sns.scatterplot(x = metadata["diet_PC1"], y = metadata["diet_PC2"], palette = custom_gradient,
                    hue = metadata["rain_month_mm"])
    plt.title("By rain")
    plt.savefig("By rain")
    plt.show()

    sns.scatterplot(x = metadata["diet_PC1"], y = metadata["diet_PC2"], palette = custom_palette[0:90:5],
                    hue = metadata["month"])
    plt.title("By month")
    plt.savefig("By month")
    plt.show()

# visualize data
def visualize_data(data):
    distance_matrix(data)


def distance_matrix(data, redo=False):
    if redo:
        d_matrix = squareform(pdist(data.iloc[:, 2:], metric = 'braycurtis'))
        d_matrix = (d_matrix)
        mds = MDS(n_components = 3)
        transformed = mds.fit_transform(d_matrix)
        pd.DataFrame(transformed).to_csv("transformed_data.csv")

    transformed = pd.read_csv("transformed_data.csv", index_col = 0)
    sns.scatterplot(x = transformed["0"], y = transformed["1"], hue = data["season"]) #, palette = custom_palette[0:90:5]
    plt.title("PCoA by Bray-Curtis Distance")
    plt.xlabel("PCoA1")
    plt.ylabel("PCoA2")

    # Plot movement arrows
    for baboon in data["baboon_id"].unique()[:1]:
        print(baboon)
        baboon_df = data[data["baboon_id"]==baboon].sort_values("collection_date")
        baboon_transformed = transformed.iloc[baboon_df.index, :].reset_index(drop=True)
        for sample_index in range(len(baboon_transformed)-1):
            x, y = baboon_transformed.iloc[sample_index,:][0], baboon_transformed.iloc[sample_index,:][1]
            dx = baboon_transformed.iloc[sample_index+1,:][0] - baboon_transformed.iloc[sample_index,:][0]
            dy = baboon_transformed.iloc[sample_index+1,:][1] - baboon_transformed.iloc[sample_index,:][1]

            plt.arrow(x, y, dx, dy, length_includes_head=True, head_width=0.2, head_length=0.2)


    plt.show()


def autocorrelation(data):
    bacterias = list(data.drop(columns=meta_features).columns)
    results_pacf = dict()
    for baboon in data["baboon_id"].unique():
        baboon_df = data[data["baboon_id"] == baboon].sort_values("collection_date")
        baboon_df = baboon_df.drop(columns=meta_features)
        for bacteria in bacterias:
            results_pacf[baboon+bacteria] = pacf(baboon_df[bacteria], nlags=10)
    pacf_df = pd.DataFrame(results_pacf).T
    for i in range(10):
        sns.boxplot(x=i, y=pacf_df[i], color=custom_palette[10*i])

    plt.title("Partial Auto-correlation Across Baboons and Bacterias")
    plt.xlabel("Lag")
    plt.ylabel("Partial Correlation")
    plt.show()

def main():
    data, metadata = load_data()
    # visualize_metadata(metadata)
    # visualize_data(data)
    # FFQ_plot(metadata)
    autocorrelation(data)


if __name__ == '__main__':
    main()
