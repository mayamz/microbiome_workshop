import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime

custom_palette = list(sns.color_palette("hls", 100))
DATA_PATH = "./data/"
meta_features = ["sample", "baboon_id", "collection_date"]


def load_data():
    metadata = pd.read_csv(f"{DATA_PATH}train_metadata.csv")
    metadata["collection_date"] = pd.to_datetime(metadata["collection_date"])
    data = pd.read_csv(f"{DATA_PATH}train_data.csv")

    species = list(data.columns)
    species.remove("sample")

    data = pd.merge(data, metadata[meta_features], on='sample', how='inner')
    data = data[meta_features + species]
    return data, metadata


def aggregate_samples(data, discretization_parameter='W'):
    cols_for_first = ['sample', "baboon_id"]
    data.set_index('collection_date', inplace=True)

    x_numerical = pd.DataFrame(data.groupby([data['baboon_id'], pd.Grouper(level='collection_date',
                                                                           freq=discretization_parameter)])
                               [[col for col in data.columns if col not in cols_for_first]].agg('mean').to_records())

    x_params = pd.DataFrame(data.groupby([data['baboon_id'], pd.Grouper(level='collection_date',
                                        freq=discretization_parameter)], as_index=False).agg('first')[cols_for_first])
    data = pd.concat([x_params[cols_for_first], x_numerical.drop(columns=['baboon_id'])], axis=1)
    data["collection_date"] = data["collection_date"].astype('<M8[ns]')

    return data


def distance_metric(x, y):
    # TODO: This is the distance metric, it will be changed
    dist = np.abs((x["collection_date"] - y["collection_date"]).total_seconds() / (60 * 60 * 24)) # time diff
    dist += (x["baboon_id"] == y["baboon_id"])  # baboon identity
    return dist


def single_knn_interpolation(data, data_to_complete, K):
    """
    Based on the KNN interpolation from "Interpolation of Microbiome Composition in Longitudinal Datasets" (Peleg & Borenstein)
    Returns an interpolation of the data point using KNN kernel with Epanechnikov function.
    """

    # Calculate distance for each point (by our distance metric), and choose K nearest
    data["distance_from_point"] = data.apply(lambda row: distance_metric(row, data_to_complete), axis=1)

    data = data.sort_values(by=["distance_from_point"]).reset_index(drop=True)
    data = data.iloc[:K, :]

    # Calculate Epanechnikov kernel function
    Kpoint = data.loc[K-1, "distance_from_point"]
    data = data[[col for col in data.columns if col not in data_to_complete.index]]
    data["K(t)"] = data["distance_from_point"] / Kpoint

    # Interpolate each taxon with the kernel
    taxa_cols = [col for col in data.columns if col not in ["distance_from_point", "K(t)"]]
    for taxon in taxa_cols:
        data[taxon] = data.apply(lambda row: 0.75 * (1 - row["K(t)"] ** 2) * row[taxon], axis=1)

    data = data.drop(columns=["distance_from_point", "K(t)"])
    time_point_interpolated = data.sum()

    # Normalize the generated sample so sum(generated_sample) = 1
    sum_time_point = sum(time_point_interpolated)
    time_point_interpolated = time_point_interpolated / sum_time_point

    time_point_interpolated = pd.concat([time_point_interpolated, data_to_complete])
    return time_point_interpolated  # TODO: Should return data_to_complete + time_point_interpolated?


def knn_interpolation(data, K=5):
    """ Detects missing dates for each baboon and interpolates by KNN """
    interpolated_df = pd.DataFrame()

    for baboon_id in data["baboon_id"].unique():
        baboon_df = data[data["baboon_id"] == baboon_id]
        baboon_min_date = baboon_df["collection_date"].min()
        baboon_max_date = baboon_df["collection_date"].max()

        baboon_df["collection_date"] = pd.to_datetime(baboon_df["collection_date"])
        for date in pd.date_range(baboon_min_date, baboon_max_date, freq="w"):
            if date not in baboon_df["collection_date"].values:
                sample_metadata = baboon_df.iloc[0, :][["baboon_id", "collection_date", "sample"]]
                sample_metadata["collection_date"] = date

                interpolated_df = pd.concat([interpolated_df, pd.DataFrame(single_knn_interpolation(data.copy(), sample_metadata, K)).T],
                                            ignore_index=True)
    data["interpolated"] = False
    interpolated_df["interpolated"] = True

    full_data = pd.concat([data, interpolated_df], ignore_index=True)
    return full_data
