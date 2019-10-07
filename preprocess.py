from scipy.special import erfinv
import numpy as np
import pandas as pd
import random
import feather
def df_concat(flist):
    add_columns = flist[0].columns.values.tolist()
    add_df = flist[0].values
    for i in flist[1:]:
        add_columns += i.columns.values.tolist()
        add_df = np.c_[add_df,i.values]

    return pd.DataFrame(add_df,columns=add_columns)

def rank_gauss(df):
    for c in df.columns:
        series = df[c].rank()
        M = series.max()
        m = series.min() 
        series = (series-m)/(M-m)
        series = series - series.mean()
        series = series.apply(erfinv) 
        df[c] = series
    return df
def Swap_noise(array):
    height = len(array)
    width = len(array[0])
    rands = np.random.uniform(0, 1, (height, width) )
    copy  = np.copy(array)

    for h in range(height):
        for w in range(width):
            if rands[h, w] <= 0.10:
                swap_target_h = random.randint(0,height)
                copy[h, w] = array[swap_target_h-1, w]
    return copy

def load_data(predict=False):
    train = feather.read_dateframe("../data/input/train.feather")
    test = feather.read_dateframe("../data/input/test.feather")
    del train["Score"]
    if predict:
        rank_data = feather.read_dataframe("../data/input/rank_gauss.feather").values
        return train,test,rank_data
    
    return train,test
def main():
    print('Transforming data')
    train,test = load_data()
    data = train.append(test)
    #only this datasets case (rank_gauss function don't apply category and binaly data)
    num_c = [ i for i in train.columns if len(data[i].unique()) != 2]
    cate_c = [ i for i in train.columns if len(data[i].unique()) == 2]
    num_all = data[num_c]
    cate_all = data[cate_c]
    rank_num = rank_gauss(num_all)
    rank_all_data = df_concat([rank_num,cate_all])
    print(rank_all_data.shape)
    rank_all_data.to_feather("../data/input/rank_gauss.feather")

if __name__ == "__main__":
    main()