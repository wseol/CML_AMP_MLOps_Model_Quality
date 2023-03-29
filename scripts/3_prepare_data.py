#!/usr/bin/env python
# coding: utf-8

# # 2. Preparing data for model training

# In[1]:


import os
import numpy as np
import pandas as pd
import pickle
import sys

from datetime import datetime
from src.utils import random_day_offset, outlier_removal
np.random.seed(42)


# In[2]:


import os
import shutil
import pandas as pd
import datetime

from impala.util import as_pandas
from joblib import dump, load


# In[3]:


import cml.data_v1 as cmldata


# In[4]:


CONNECTION_NAME = "default-hive-aws"

## Alternate Sample Usage to provide different credentials as optional parameters
conn = cmldata.get_connection(
   CONNECTION_NAME, {"USERNAME": "wseol", "PASSWORD": "Workload_Password1"}
)

## Alternate Sample Usage to get DB API Connection interface
db_conn = conn.get_base_connection()

## Alternate Sample Usage to get DB API Cursor interface
EXAMPLE_SQL_QUERY = "select * from airlines.flights_wseol"

db_cursor = conn.get_cursor()
db_cursor.execute(EXAMPLE_SQL_QUERY)


# In[5]:


df = as_pandas(db_cursor)


# In[6]:


len(df)


# In[7]:


df.columns = df.columns.str.replace('flights_wseol.','')


# In[8]:


df.rename(columns = {'cancelled':'TARGET'}, inplace = True)


# In[9]:


df.head(3)


# In[10]:


numerical_cols = [i for i in df.columns if df.dtypes[i]!='object']

# select features only from columns
features = numerical_cols
features.remove('TARGET')


# In[11]:


# save features name for the next tasks
outputFile = open('/home/cdsw/features.pkl', 'wb')
pickle.dump(features, outputFile)
outputFile.close()


# In[12]:


# # change extraordinary data to avoid exceptions
df.replace(np.nan, -999, inplace=True)
df.replace(np.inf, 99999, inplace=True)
df.replace(-np.inf, -99999, inplace=True)


# In[13]:


# select only the columns that using
df = df[features+['TARGET']]


# In[14]:


df.head(3)


# ## Dividing data to simulate data drift

# In[15]:


df["id"] = np.random.randint(1, 1000000000, size=len(df))


# In[16]:


df["date_departure"] = np.random.choice(pd.date_range('2014-05-02', '2015-05-27'), len(df))
df["date_forecast"] = df.date_departure.apply(lambda x: random_day_offset(x))


# In[17]:


# Split out first 6 months of data for training, remaining for simulating a "production" scenario
min_repay_date = df.date_departure.min()
max_repay_date = (
    df.date_departure.max().to_period("M").to_timestamp()  # drop the partial last month
)


# In[18]:


train_df = df[
    df.date_departure.between(min_repay_date, "2014-10-31", inclusive="both")
].sort_values("date_departure")


# In[19]:


prod_df = df[
    df.date_departure.between("2014-10-31", max_repay_date, inclusive=False)
].sort_values("date_departure")


# ## saving processed data

# In[20]:


working_dir = "/home/cdsw/data/working/"
os.makedirs(working_dir, exist_ok=True)
dfs = [("train", train_df), ("prod", prod_df)]
for name, dataframe in dfs:
    path = os.path.join(working_dir, f"{name}_df.pkl")
    dataframe.to_pickle(path)


# In[ ]:




