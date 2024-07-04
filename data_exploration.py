import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "./data/"

metadata = pd.read_csv(f"{DATA_PATH}train_metadata.csv")
data = pd.read_csv(f"{DATA_PATH}train_data.csv")