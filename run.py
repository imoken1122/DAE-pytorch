import torch as th
from torch import nn,optim
from torch.autograd import Variable as V
from torch.functional import F
import pandas as pd
import numpy as np
import feather
from preprocess import Swap_noise
from torch.optim.lr_scheduler import  LambdaLR
from dae import DAE


def create_batch1(x,y,batch_size,shuffle):
    if shuffle:
        a = list(range(len(x)))
        np.random.shuffle(a)
        x = x[a]
        y = y[a]

    batch_x = [x[batch_size * i : (i+1)*batch_size,:].tolist() for i in range(len(x)//batch_size)]
    batch_y = [y[batch_size * i : (i+1)*batch_size,:].tolist() for i in range(len(x)//batch_size)]
    return np.array(batch_x), np.array(batch_y)

def train_(trainx,trainy,opt,loss_f,batch_size):
    model_dae.train()
    batch_x,batch_y = create_batch1(trainx,trainy,batch_size,True)
    run_loss,pred = 0.,0
    for x,y in zip(batch_x,batch_y):
        opt.zero_grad()
        x,y = V(th.Tensor(x)),V(th.Tensor(y))
        output = model_dae(x,False)
        loss = loss_f(output,y)
        loss.backward()
        opt.step()
        run_loss += loss.item()
    
    return run_loss/(len(trainx)//batch_size)

def valid_(valx,valy,opt,loss_f,batch_size):
    model_dae.eval()

    batch_x,batch_y = create_batch1(valx,valy, batch_size,False)
    run_loss = 0.
    for x,y in zip(batch_x,batch_y):

        x,y = th.Tensor(x),th.Tensor(y)
        output = model_dae(x,False)

        loss = loss_f(output,y)
        run_loss += loss.item()

    return run_loss/(len(valx)//batch_size)

if __name__ == "__main__":

    df = feather.read_dataframe("../data/input/rank_gauss.feather").values
    tr = feather.read_dataframe("../data/input/train").values
    n_train = len(tr); del tr
    DECAY = 0.95
    BATHCSIZE = 128
    EPOCH = 2
    CYCLE = 200

    input_dim = df.shape[1]
    output_dim = df.shape[1]
    model_dae = DAE(input_dim,output_dim)
    opt = optim.Adam(model_dae.parameters())
    loss_f = nn.MSELoss()
    scheduler = LambdaLR(opt, lr_lambda = lambda i : DECAY**i)


    tr_losses,val_losses = [],[]
    tr_r2,te_r2 = [],[]

    for i in range(CYCLE):
        all_noise = Swap_noise(df)
        all_org = df
        test_data_noise,test_data_org = all_noise[n_train:,:], all_org[n_train:,:]

        for e in range(EPOCH):
            tr_loss = train_(all_noise,all_org,opt,loss_f,BATHCSIZE)
            val_loss = valid_(test_data_noise,test_data_org,opt,loss_f,BATHCSIZE)
            tr_losses.append(tr_loss)
            val_losses.append(val_loss)
        lr = opt.state_dict()["param_groups"][0]["lr"]
        if i > 100 or i % 20 == 0:    
            scheduler.step()
        print(f"cycle : {i} \t trloss : {tr_loss} \t valloss : {val_loss} \t lr :{lr}")
    
        th.save(model_dae.state_dict(), f'./save_model/{i}_{tr_loss}.pth')
