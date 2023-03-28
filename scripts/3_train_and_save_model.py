#!/usr/bin/env python
# coding: utf-8

# # 3. Training model

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

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,train_test_split
from sklearn.metrics import classification_report


# In[2]:


current_time = datetime.datetime.now()
print ("Time now at greenwich meridian is : ", end = "")
print (current_time)


# ## preparing data

# In[3]:


# read all features
file = open("/home/cdsw/features.pkl",'rb')
features = pickle.load(file)
file.close()


# In[4]:


# read training data
train_path = "/home/cdsw/data/working/train_df.pkl"
cancelled_flights = pd.read_pickle(train_path)


# In[5]:


X = cancelled_flights[features]


# In[6]:


y = cancelled_flights[["TARGET"]]


# In[7]:


# there is no categorical column at this point
categorical_cols = []
# categorical_cols = [i for i in X.columns if X.dtypes[i]=='object']

ct = ColumnTransformer(
    [("le", OneHotEncoder(), categorical_cols)], remainder="passthrough"
)
X_trans = ct.fit_transform(X)


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X_trans, y, random_state=42)


# ## Defining model

# In[9]:


get_ipython().system('pip install xgboost')


# In[10]:


get_ipython().system('pip install hyperopt')


# In[11]:


import time
import warnings
import numpy as np
import xgboost as xgb

#hyperparameter tuning
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK
from hyperopt.pyll import scope
from sklearn.metrics import roc_auc_score


# In[12]:


# using XGBoost
default_params = {}
xgbclf = xgb.XGBClassifier()
gparams = xgbclf.get_params()

#default parameters have to be wrapped in lists - even single values - so GridSearchCV can take them as inputs
for key in gparams.keys():
    gp = gparams[key]
    default_params[key] = [gp]

# Create XGBoost DMatrix objects for efficient data handling

train = xgb.DMatrix(data=X_train, label=y_train)
test = xgb.DMatrix(data=X_test, label=y_test)
    
#list of hyperparameters available to Tune
default_params


# In[13]:


search_space = {
    'learning_rate': hp.loguniform('learning_rate', -7, 0),
    'max_depth': scope.int(hp.uniform('max_depth', 1, 100)),
    'min_child_weight': hp.loguniform('min_child_weight', -2, 3),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'gamma': hp.loguniform('gamma', -10, 10),
    'alpha': hp.loguniform('alpha', -10, 10),
    'lambda': hp.loguniform('lambda', -10, 10),
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'seed': 123,
}


# In[14]:


# With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
def train_model(params):
    mlflow.xgboost.autolog(silent=True)

    # However, we can log additional information by using an MLFlow tracking context manager
    with mlflow.start_run(nested=True):

        # Train model and record run time
        start_time = time.time()
        booster = xgb.train(params=params, dtrain=train, num_boost_round=10, evals=[(test, "test")], early_stopping_rounds=3, verbose_eval=False)
        run_time = time.time() - start_time
        mlflow.log_metric('runtime', run_time)

        # Record AUC as primary loss for Hyperopt to minimize
        predictions_test = booster.predict(test)
        auc_score = roc_auc_score(y_test, predictions_test)
        mlflow.log_metric('test-auc', auc_score)

        # Set the loss to -1*auc_score so fmin maximizes the auc_score
        return {'status': STATUS_OK, 'loss': -auc_score, 'booster': booster.attributes()}


# ## Training model with experiments

# In[15]:


#spark_trials = SparkTrials(parallelism=4)

#http://hyperopt.github.io/hyperopt/scaleout/spark/ Working with SPARK trials

mlflow.set_experiment('HPsearch ' + str(current_time))
with mlflow.start_run(run_name='initial_search'):
    best_params = fmin(
      fn=train_model,
      space=search_space,
      algo=tpe.suggest,
      max_evals=15,
      rstate=np.random.default_rng(123),
      #trials=spark_trials
    )


# In[16]:


print(best_params)


# In[17]:


best_params['max_depth']=int(best_params['max_depth'])


# ## Training the best model

# In[18]:


xgbclf_base = xgb.XGBClassifier(**best_params, n_estimators=10)
pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("xgbclf_base", xgbclf_base)])
pipe.fit(X_train, y_train)


# In[19]:


# create classification report
y_pred = pipe.predict(X_test)
targets = ["Not-cancelled", "Cancelled"]
cls_report = classification_report(y_test, y_pred, target_names=targets)
print(cls_report)


# In[20]:


print ('ROC AUC Score',roc_auc_score(y_test,y_pred))


# ## Saving the model pipeline

# In[21]:


dump(ct, "/home/cdsw/ct.joblib")


# In[22]:


dump(pipe, "/home/cdsw/pipe.joblib")


# In[ ]:




