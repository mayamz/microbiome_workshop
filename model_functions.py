import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LinearRegression


custom_palette = list(sns.color_palette("hls", 100))
DATA_PATH = "./data/"
meta_features = ["sample", "baboon_id", "collection_date",
                 "age", "sex", "social_group",
                 "month", "rain_month_mm",
                 "diet_PC1", "diet_PC2"]


def load_data():
    metadata = pd.read_csv(f"{DATA_PATH}train_metadata.csv")
    metadata["collection_date"] = pd.to_datetime(metadata["collection_date"])
    data = pd.read_csv(f"{DATA_PATH}train_data.csv")

    species = list(data.columns)
    species.remove("sample")

    data = pd.merge(data, metadata[meta_features], on = 'sample', how = 'inner')
    data = data[meta_features + species]

    # Convert month to cyclic
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12.0)

    return data, metadata


def aggregate_samples(data, discretization_parameter='W'):
    cols_for_first = [col for col in meta_features if col != "collection_date"]
    data.set_index('collection_date', inplace = True)

    x_numerical = pd.DataFrame(data.groupby([data['baboon_id'], pd.Grouper(level = 'collection_date',
                                                                           freq = discretization_parameter)])
                               [[col for col in data.columns if col not in cols_for_first]].agg('mean').to_records())

    x_params = pd.DataFrame(data.groupby([data['baboon_id'], pd.Grouper(level = 'collection_date',
                                                                        freq = discretization_parameter)],
                                         as_index = False).agg('first')[cols_for_first])
    data = pd.concat([x_params[cols_for_first], x_numerical.drop(columns = ['baboon_id'])], axis = 1)
    data["collection_date"] = data["collection_date"].astype('<M8[ns]')

    return data


def interpolation_dist_metric(x, y):
    # TODO: This is the interpolation distance metric, it will be changed
    dist = np.abs((x["collection_date"] - y["collection_date"]).total_seconds() / (60 * 60 * 24))  # time diff
    dist += (x["baboon_id"] == y["baboon_id"])  # baboon identity
    return dist


def seasonal_dist_metric(x, y):
    # TODO: This is the seasonal KNN model distance metric, it will be changed
    # representing the closeness to the sample's baboon - social_group, age, sex
    identity_metric = (x["social_group"] == y["social_group"])  # group identity
    identity_metric += np.abs(x["age"] - y["age"])  # age similarity
    identity_metric += (x["sex"] == y["sex"])  # sex identity

    # representing the closeness to the sample's season - month, rain_month_mm
    time_metric = np.linalg.norm(x[["month_sin", "month_cos"]], y[["month_sin", "month_cos"]])  # month similarity
    time_metric += np.abs(x["rain_month_mm"] - y["rain_month_mm"])  # rain similarity

    # representing the closeness to the sample's diet  -  diet_PC1,diet_PC2
    diet_metric = np.abs(x["diet_PC1"] - y["diet_PC1"])  # diet_PC1
    diet_metric += 0.5 * np.abs(x["diet_PC2"] - y["diet_PC2"])  # diet_PC2

    dist = identity_metric + time_metric + diet_metric
    return dist


def single_knn_interpolation(data, data_to_complete, K, distance_metric):
    """
    Based on the KNN interpolation from "Interpolation of Microbiome Composition in Longitudinal Datasets" (Peleg & Borenstein)
    Returns an interpolation of the data point using KNN kernel with Epanechnikov function.
    """

    # Calculate distance for each point (by our distance metric), and choose K nearest
    data["distance_from_point"] = data.apply(lambda row: distance_metric(row, data_to_complete), axis = 1)

    data = data.sort_values(by = ["distance_from_point"]).reset_index(drop = True)
    data = data.iloc[:K, :]

    # Calculate Epanechnikov kernel function
    Kpoint = data.loc[K - 1, "distance_from_point"]
    data = data[[col for col in data.columns if col not in data_to_complete.index]]
    data["K(t)"] = data["distance_from_point"] / Kpoint

    # Interpolate each taxon with the kernel
    taxa_cols = [col for col in data.columns if col not in ["distance_from_point", "K(t)"]]
    for taxon in taxa_cols:
        data[taxon] = data.apply(lambda row: 0.75 * (1 - row["K(t)"] ** 2) * row[taxon], axis = 1)

    data = data.drop(columns = ["distance_from_point", "K(t)"])
    time_point_interpolated = data.sum()

    # Normalize the generated sample so sum(generated_sample) = 1
    sum_time_point = sum(time_point_interpolated)
    time_point_interpolated = time_point_interpolated / sum_time_point

    time_point_interpolated = pd.concat([time_point_interpolated, data_to_complete])
    return time_point_interpolated


def knn_interpolation(data, K=5):
    """ Detects missing dates for each baboon and interpolates by KNN """
    interpolated_df = pd.DataFrame()
    i = 0
    for baboon_id in data["baboon_id"].unique():
        baboon_df = data[data["baboon_id"] == baboon_id]
        baboon_min_date = baboon_df["collection_date"].min()
        baboon_max_date = baboon_df["collection_date"].max()

        baboon_df["collection_date"] = pd.to_datetime(baboon_df["collection_date"])
        for date in pd.date_range(baboon_min_date, baboon_max_date, freq = "w"):
            if date not in baboon_df["collection_date"].values:
                sample_metadata = baboon_df.iloc[0, :][meta_features]
                sample_metadata["collection_date"] = date

                interpolated_df = pd.concat([interpolated_df, pd.DataFrame(
                    single_knn_interpolation(data.copy(), sample_metadata, K, interpolation_dist_metric)).T],
                                            ignore_index = True)
        i += 1
        print(i)
    data["interpolated"] = False
    interpolated_df["interpolated"] = True

    full_data = pd.concat([data, interpolated_df], ignore_index = True)
    return full_data


def seasonal_pred(data, x_test, K=5):
    """ Predict the seasonal effect """
    x_pred = pd.DataFrame()
    for index, test_row in x_test.iterrows():
        row_pred = pd.DataFrame(
            single_knn_interpolation(data.copy(), test_row, K, distance_metric = seasonal_dist_metric))
        x_pred = pd.concat([x_pred, row_pred.T], ignore_index=True)
    return x_pred


def linear_reg_per_baboon(data, x_test, baboon_id):
    baboon_data = data[data["baboon_id"] == baboon_id]
    taxa_columns = [col for col in baboon_data.columns if col not in x_test.columns]

    dates_to_pred = x_test[x_test["baboon_id"] == baboon_id]["collection_date_number"].to_numpy().reshape(-1, 1)

    model = LinearRegression()
    model.fit(baboon_data["collection_date_number"].to_numpy().reshape(-1, 1), baboon_data[taxa_columns].to_numpy())
    y_pred = pd.DataFrame(model.predict(dates_to_pred), columns=taxa_columns)
    x_test = x_test.reset_index(drop=True)
    y_pred = pd.concat([x_test, y_pred], axis=1)

    # normalize
    y_pred[taxa_columns] = y_pred[taxa_columns] + np.abs(y_pred[taxa_columns].min())
    y_pred[taxa_columns] = y_pred[taxa_columns].div(y_pred[taxa_columns].sum(axis=1), axis=0)
    return y_pred


def trend_pred(data, x_test):
    """ Predict the trend per baboon"""
    # add another column for ordinal date
    min_date = data["collection_date"].min().toordinal()
    data["collection_date_number"] = data["collection_date"].apply(lambda x: (x.toordinal() - min_date))
    x_test["collection_date_number"] = x_test["collection_date"].apply(lambda x: (x.toordinal() - min_date))

    x_pred = pd.DataFrame()

    for baboon_id in x_test["baboon_id"].unique():
        baboon_pred = linear_reg_per_baboon(data, x_test, baboon_id)
        x_pred = pd.concat([x_pred, baboon_pred], ignore_index=True)
    return x_pred


def predict(data, x_test):
    """ Predict microbiome for x_test metadata using a hybrid of two methods

    :param data:    microbiome int
    :param x_test:
    :return:
    """
    # Get prediction from both models
    seasonal_prediction = seasonal_pred(data, x_test)
    trend_prediction = trend_pred(data, x_test)

    taxa_cols = [col for col in seasonal_prediction.columns if col not in x_test.columns]

    # Merge the two models by averaging them
    x_pred = seasonal_prediction.copy()
    x_pred[taxa_cols] = seasonal_prediction[taxa_cols] * 0.5 + trend_prediction[taxa_cols] * 0.5

    return x_pred
