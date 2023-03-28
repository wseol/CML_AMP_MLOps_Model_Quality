#!/usr/bin/env python
# coding: utf-8

# # 5. Adding noise to data
# ### for mimicking Data Drift and Model Decay

# In[1]:


import os
import scipy
import pickle
import numpy as np
import pandas as pd
import mlflow
import argparse
import datetime

from joblib import dump, load


# In[2]:


# read all features
file = open("/home/cdsw/features.pkl",'rb')
features = pickle.load(file)
file.close()


# In[3]:


# read production data
prod_path = "/home/cdsw/data/working/prod_df.pkl"
prod_df = pd.read_pickle(prod_path)


# In[4]:


# adding noise for simulating data drift. It adds more noise over time
for col in prod_df.columns:
    if str(prod_df[col].dtypes) == 'float64' and col != 'TARGET':        
        noise = np.random.normal(0, .1, prod_df[col].shape[0]) * prod_df[col].iloc[0] * (1/1000) * (prod_df.date_departure - pd.to_datetime('2014-10-31', format='%Y-%m-%d')).dt.days * (prod_df.date_departure - pd.to_datetime('2014-10-31', format='%Y-%m-%d')).dt.days
        prod_df[col] = prod_df[col] + round(noise,3)


# In[5]:


# adding noise for simulating data drift. It adds more noise over time
for col in prod_df.columns:
    if str(prod_df[col].dtypes) == 'int64' and col != 'TARGET':        
        noise = np.random.normal(0, .1, prod_df[col].shape[0]) * prod_df[col].iloc[0] * (1/1000) * (prod_df.date_departure - pd.to_datetime('2014-10-31', format='%Y-%m-%d')).dt.days * (prod_df.date_departure - pd.to_datetime('2014-10-31', format='%Y-%m-%d')).dt.days
        prod_df[col] = prod_df[col] + round(noise, 0).astype(int)


# In[ ]:


# adding noise for simulating target drift. It adds more noise over time
def categorise(row): 
    noise = np.random.uniform(0, 1) * (1/1000) * (row.date_departure - pd.to_datetime('2014-10-31', format='%Y-%m-%d')).days * (row.date_departure - pd.to_datetime('2014-10-31', format='%Y-%m-%d')).days
    if noise > 0.5:
        if row.TARGET == 1:
            row.TARGET = 0
        else:
            row.TARGET = 1
    return row.TARGET
prod_df['TARGET'] = prod_df.apply(lambda row: categorise(row), axis=1)


# ## saving processed data

# In[ ]:


path = os.path.join("/home/cdsw/data/working/prod_df.pkl")
prod_df.to_pickle(path)


# In[ ]:




