import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression


custom_palette = list(sns.color_palette("hls", 100))
DATA_PATH = "./data/"
meta_features = ["sample", "baboon_id", "collection_date",
                 "age", "sex", "social_group",
                 "month", "rain_month_mm",
                 "diet_PC1", "diet_PC2", "diet_PC3"]


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

def change_date(date):
    day_of_week = date.weekday()
    if day_of_week < 3 or day_of_week == 6:  # Sunday, Monday, Tuesday, Wednesday
        return date
    return date + pd.DateOffset(days=3)


def aggregate_samples(data, discretization_parameter='W'):
    cols_for_first = [col for col in meta_features if col != "collection_date"]

    data["collection_date"] = data["collection_date"].apply(lambda x: change_date(x))
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
    # weights - can be learned
    time_weight = 0.01
    social_weight = 0.6
    baboon_weight = 0.2
    age_weight = 0.25
    sex_weight = 0.25
    group_weight = 0.5

    # representing the closeness to the sample's time, in days
    time_dist = np.abs((x["collection_date"] - y["collection_date"])).total_seconds() / (60 * 60 * 24)  # time diff

    # representing the closeness to the sample's baboon - social_group, age, sex
    age_dist = age_weight * (np.abs((x["age"] - y["age"])) / 5)
    sex_dist = sex_weight * (x["sex"] != y["sex"])
    group_dist = group_weight * (x["social_group"] != y["social_group"])  # same social group - 0, else 1

    baboon_dist = (x["baboon_id"] != y["baboon_id"])  # same baboon - 0, else 1

    dist = time_weight * time_dist + social_weight * (age_dist + sex_dist + group_dist) + baboon_weight * baboon_dist
    return dist


def seasonal_dist_metric(x, y):
    # weights - can be learned
    season_weight = 0.4
    identity_weight = 1
    diet_weight = 0.018
    age_weight = 0.25
    sex_weight = 0.25
    group_weight = 0.5
    month_weight = 0.5
    rain_weight = 0.5
    PC1_weight = 4/7
    PC2_weight = 2/7
    PC3_weight = 1/7

    # representing the closeness to the sample's baboon - social_group, age, sex
    group_dist = group_weight * (x["social_group"] != y["social_group"])  # group identity
    age_dist = age_weight * (np.abs(x["age"] - y["age"])) / 5  # age similarity
    sex_dist = sex_weight * (x["sex"] != y["sex"])  # sex identity

    # representing the closeness to the sample's season - month, rain_month_mm
    month_dist = month_weight * (np.linalg.norm(x[["month_sin", "month_cos"]] - y[["month_sin", "month_cos"]])) / 2  # month similarity
    rain_dist = rain_weight * (np.abs(x["rain_month_mm"] - y["rain_month_mm"]) / 20)  # rain similarity

    # representing the closeness to the sample's diet  -  diet_PC1, diet_PC2, diet_PC3
    diet_metric = PC1_weight * np.abs(x["diet_PC1"] - y["diet_PC1"])   # diet_PC1
    diet_metric += PC2_weight * np.abs(x["diet_PC2"] - y["diet_PC2"])  # diet_PC2
    diet_metric += PC3_weight * np.abs(x["diet_PC3"] - y["diet_PC3"])  # diet_PC3

    dist = identity_weight * (age_dist + sex_dist + group_dist) + season_weight * (month_dist + rain_dist) + diet_weight * diet_metric
    return dist


def single_knn_interpolation(data, data_to_complete, K, distance_metric, filter_dates=False):
    """
    Based on the KNN interpolation from "Interpolation of Microbiome Composition in Longitudinal Datasets" (Peleg & Borenstein)
    Returns an interpolation of the data point using KNN kernel with Epanechnikov function.
    """
    # Filter near timepoints
    if filter_dates:
        data = data[np.abs((data["collection_date"] - data_to_complete["collection_date"])).dt.total_seconds() / (60 * 60 * 24) < 110]

    # Calculate distance for each point (by our distance metric), and choose K nearest
    data["distance_from_point"] = data.apply(lambda row: distance_metric(row, data_to_complete), axis = 1)
    data = data.sort_values(by = ["distance_from_point"]).reset_index(drop = True)
    data = data.iloc[:K, :]

    # Calculate Epanechnikov kernel function
    Kpoint = data.loc[min(len(data) - 1, K - 1), "distance_from_point"]
    data = data[[col for col in data.columns if col not in data_to_complete.index]]
    data["K(t)"] = data["distance_from_point"] / Kpoint

    # Interpolate each taxon with the kernel
    taxa_cols = [col for col in data.columns if col not in ["distance_from_point", "K(t)"]]
    for taxon in taxa_cols:
        data[taxon] = data.apply(lambda row: 0.75 * (1 - row["K(t)"] ** 2) * row[taxon], axis = 1)

    data = data.drop(columns = ["distance_from_point", "K(t)"])
    time_point_interpolated = data.sum()

    # Normalize the generated sample so sum(generated_sample) = 1
    time_point_interpolated = time_point_interpolated + np.abs(time_point_interpolated.min())
    sum_time_point = sum(time_point_interpolated)
    time_point_interpolated = time_point_interpolated / sum_time_point

    time_point_interpolated = pd.concat([data_to_complete, time_point_interpolated])
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
                    single_knn_interpolation(data.copy(), sample_metadata, K, interpolation_dist_metric, filter_dates=True)).T], ignore_index = True)

        i += 1
        print(f"finished interpolating baboon {i}")
    data["interpolated"] = False
    interpolated_df["interpolated"] = True

    full_data = pd.concat([data, interpolated_df], ignore_index = True)
    return full_data


def seasonal_pred(data, x_test, K=5):
    """ Predict the seasonal effect """
    x_pred = pd.DataFrame()
    i = 0
    for index, test_row in x_test.iterrows():
        row_pred = pd.DataFrame(single_knn_interpolation(data.copy(), test_row, K, distance_metric = seasonal_dist_metric))
        x_pred = pd.concat([x_pred, row_pred.T], ignore_index=True)
        i += 1
        if i % 50 == 0:
            print(f"finished {i} out of {len(x_test)}")
    return x_pred


def linear_reg_per_baboon(data, x_test, baboon_id):
    baboon_data = data[data["baboon_id"] == baboon_id]
    taxa_columns = [col for col in baboon_data.columns if col not in x_test.columns]

    # fit only in last samples
    baboon_data["collection_date_number"] = baboon_data["collection_date_number"].sort_values()
    baboon_data = baboon_data.iloc[-min(len(baboon_data) + 1, 30):, :]

    dates_to_pred = x_test["collection_date_number"].to_numpy().reshape(-1, 1)

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
        baboon_pred = linear_reg_per_baboon(data, x_test[x_test["baboon_id"] == baboon_id], baboon_id)
        x_pred = pd.concat([x_pred, baboon_pred], ignore_index=True)
    return x_pred


def predict(data, x_test):
    """ Predict microbiome for x_test metadata using a hybrid of two methods"""
    x_test = x_test.reset_index()

    # Get prediction from both models
    print("start seasonal_pred")
    seasonal_prediction = seasonal_pred(data, x_test)
    seasonal_prediction = seasonal_prediction.sort_values("index")
    print("finished seasonal_pred\nstart trend_pred")
    trend_prediction = trend_pred(data, x_test)
    trend_prediction = trend_prediction.sort_values("index")
    print("finished trend_pred")
    taxa_cols = [col for col in seasonal_prediction.columns if col not in x_test.columns]

    # Merge the two models by averaging them
    x_pred = seasonal_prediction.copy()
    x_pred[taxa_cols] = seasonal_prediction[taxa_cols] * 0.5 + trend_prediction[taxa_cols] * 0.5

    return x_pred
