#!/usr/bin/env python
# coding: utf-8

# # 5. Simulating data drift

# In[1]:


import os
import cdsw
import logging
import numpy as np
import pandas as pd
import pickle
import sys
sys.path.append('/home/cdsw')

from typing import Dict
from tqdm import tqdm
from pandas.tseries.offsets import DateOffset
from evidently.dashboard import Dashboard
from evidently.tabs import (
    DataDriftTab,
    NumTargetDriftTab,
    RegressionPerformanceTab,
)

from src.utils import scale_prices
from src.api import ApiUtility
from src.inference import ThreadedModelRequest
from src.simulation import Simulation


# In[2]:


file = open("/home/cdsw/features.pkl",'rb')
features = pickle.load(file)
file.close()


# In[3]:


train_path = "/home/cdsw/data/working/train_df.pkl"
prod_path = "/home/cdsw/data/working/prod_df.pkl"


# In[4]:


train_df = pd.read_pickle(train_path)
prod_df = pd.read_pickle(prod_path)


# In[5]:


train_df = train_df[features + ['id'] + ['date_forecast'] + ['date_departure'] + ['TARGET']]
prod_df = prod_df[features + ['id'] + ['date_forecast'] + ['date_departure'] + ['TARGET']]


# In[6]:


sim = Simulation(
    model_name="test5", dev_mode='TRUE'
)


# ## run simulation (application)
# calling the model with previous & current data, and then compare the results for detecting data drift

# In[7]:


sim.run_simulation(train_df, prod_df)

