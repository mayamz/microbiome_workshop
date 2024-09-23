
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy

from model_functions import *

from sklearn.neighbors import KNeighborsRegressor
from statsmodels.tsa.stattools import adfuller
from darts.models.forecasting.varima import VARIMA
from darts import TimeSeries
from datetime import datetime
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.statespace.varmax import VARMAX


import warnings
warnings.filterwarnings('ignore')


# ## Data loading

# In[3]:


data, metadata = load_data()


# ## Data preprocessing

# ### Split to train-test

# In[5]:


# aggregate to one sample per week
data = aggregate_samples(data.copy())


# ### Interpolation

# In[6]:


interpolated_df = knn_interpolation(data[data["baboon_id"]=="Baboon_201"])


# In[7]:


interpolated_df["collection_date"] = pd.to_datetime(interpolated_df["collection_date"])


# ### Format dataset

# In[ ]:





# ### Add random noise for non-singularity

# In[9]:


taxa_columns = [col for col in interpolated_df.columns if col not in ["sample", "baboon_id", "collection_date", "interpolated"]]
min_value = interpolated_df[taxa_columns][interpolated_df[taxa_columns] > 0].min().min()
noise = np.random.uniform(0.25 * min_value, min_value, size=interpolated_df[taxa_columns].shape)


# In[10]:


noise_df = interpolated_df.copy()
noise_df[taxa_columns] = interpolated_df[taxa_columns] + noise
#noise_df[taxa_columns] = noise_df[taxa_columns].div(noise_df[taxa_columns].sum(axis=1), axis=0)


# ## Train

# ### VARIMA - model per baboon

# In[13]:


# baboon_models = {}

# # Train a model per baboon
# for baboon in noise_df["baboon_id"].unique():
#     # Create a time series per baboon
#     baboon_data = noise_df[noise_df["baboon_id"]==baboon].drop(columns = ["sample", "baboon_id", "interpolated"])
#     baboon_data = TimeSeries.from_dataframe(baboon_data, time_col="collection_date")

#     #print(baboon_data)
    
#     # Train a VARIMA model for the baboon
#     model = VARIMA(p=1, q=1, d=1)  # TODO: handle params
#     model.fit(baboon_data)
    
#     baboon_models[baboon] = model


# In[ ]:





# In[ ]:


baboon_models, baboon_models_fitted = {}, {}

# Train a model per baboon
for baboon in noise_df["baboon_id"].unique():
    # Create a time series per baboon
    baboon_data = noise_df[noise_df["baboon_id"] == baboon].drop(columns=["sample", "baboon_id", "interpolated"])
    
    # Ensure the 'collection_date' is set as the index
    baboon_data = baboon_data.set_index('collection_date')
    baboon_data = baboon_data.apply(pd.to_numeric, errors='coerce')
    baboon_data = baboon_data.dropna()

    # Train a VARMAX model for the baboon
    model = VARMAX(baboon_data, order=(1, 1, 1), enforce_stationarity=False, initialization='approximate_diffuse')
    model_fitted = model.fit(disp=False)

    # Store the fitted model for the baboon
    baboon_models_fitted[baboon] = model_fitted
    baboon_models[baboon] = model


