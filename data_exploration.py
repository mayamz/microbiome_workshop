import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

DATA_PATH = "./data/"

def load_data():
    metadata = pd.read_csv(f"{DATA_PATH}train_metadata.csv")
    data = pd.read_csv(f"{DATA_PATH}train_data.csv")
    return data, metadata

def visualize_metadata(metadata):
    # samples per baboon df
    sample_per_baboon = metadata.groupby("baboon_id")[["sample"]].count()
    sample_per_baboon.index = sample_per_baboon.index.str.replace("Baboon_", "")
    sample_per_baboon = sample_per_baboon.sort_values(by="sample", ascending = False)

    # Hist
    # sns.histplot(sample_per_baboon, bins=60)
    # plt.title("Sample per Baboon")
    # plt.xlabel("Number of Samples per Baboon")
    # plt.ylabel("Number of Baboons")
    # plt.show()

    # sample per baboon dot plot
    # fig = plt.figure(figsize = (15,10))
    # plt.scatter(sample_per_baboon.index, sample_per_baboon["sample"])
    # plt.xticks(rotation=45, size=7)
    # plt.title("Number of Samples per Baboon ID")
    # plt.xlabel("Baboon ID")
    # plt.ylabel("Number of Samples")
    # plt.show()

    # plot diff per baboon
    # fig = plt.figure(figsize = (15, 10))
    # metadata["collection_date"] = pd.to_datetime(metadata["collection_date"])
    # for baboon in metadata["baboon_id"].unique():
    #     baboon_samples = metadata[metadata["baboon_id"]==baboon].sort_values(by="collection_date")
    #     baboon_samples['difference'] = baboon_samples["collection_date"].diff().dt.days
    #     sns.histplot(baboon_samples['difference'], label=baboon)
    # plt.title("Time Difference Between Sequential Samples")
    # plt.xlabel("Days Between Sequential Samples")
    # plt.ylabel("Samples")
    # plt.legend(ncol=4)
    # plt.show()

    # plot total diff per baboon
    fig = plt.figure(figsize = (15, 10))
    metadata["collection_date"] = pd.to_datetime(metadata["collection_date"])
    diffs = pd.Series()
    for baboon in metadata["baboon_id"].unique():
        baboon_samples = metadata[metadata["baboon_id"] == baboon].sort_values(by = "collection_date")
        baboon_samples['difference'] = baboon_samples["collection_date"].diff().dt.days
        diffs.append(baboon_samples['difference'])

    sns.histplot(diffs)
    plt.title("Time Difference Between Sequential Samples")
    plt.xlabel("Days Between Sequential Samples")
    plt.ylabel("Samples")
    plt.show()


def main():
    data, metadata = load_data()
    visualize_metadata(metadata)

if __name__ == '__main__':
    main()