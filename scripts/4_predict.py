import os
import numpy as np
import pandas as pd
import cdsw
import json
import pickle

from joblib import dump, load
from src.utils import col_order

file = open("/home/cdsw/features.pkl",'rb')
features = pickle.load(file)
file.close()

ct = load("/home/cdsw/ct.joblib")
pipe = load("/home/cdsw/pipe.joblib")

@cdsw.model_metrics
def predict(data_input):

    # Convert dict representation back to dataframe for inference
    df = pd.DataFrame.from_records([data_input["record"]]).iloc[:1,:]
    
    input_df = df[features]
    input_transformed = ct.transform(input_df)

    probas = pipe.predict_proba(input_transformed)
    prediction = np.argmax(probas)
#    proba = round(probas[0][prediction], 2)
    
    cdsw.track_metric("input_features", input_df.to_dict(orient="records")[0])
    cdsw.track_metric("predicted_result", int(prediction))

    return int(prediction)