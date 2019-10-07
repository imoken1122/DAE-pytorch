import torch as th
from dae import DAE
import feather
import numpy as np
import pandas as pd
from preprocess import load_data,df_concat


def load_weight(model):
    param = th.load("save_model/",map_location=lambda x,y:x)
    model.load_state_dict(param)

def create_DAEdata(model,df):
    model.eval()
    dae_data = model_dae(th.Tensor(df),predict = True)
    return dae_data

train,test,rank_data = load_data(True)
ntrain = len(train)
raw = train.append(test).values

input_dim = rank_data.shape[1]
output_dim = rank_data.shape[1]
model_dae = DAE(input_dim,output_dim)
load_weight(model_dae)

dae_data = create_DAEdata(model_dae, rank_data)
combi_data = np.c_[raw,dae_data]
col = [f"col_{i}" for i in range(combi_data.shape[1])]
df= pd.DataFrame(combi_data,columns=col)
tr_df,te_df= df[:ntrain,:],df[ntrain:,:]

tr_df.to_feather("data/input/train_DAE.feather")
te_df.to_feather("data/input/test_DAE.feather")
