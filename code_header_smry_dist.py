import sqlalchemy
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import datetime as dt
import os
from os import listdir
from sklearn import linear_model
from pyarrow import parquet as pq
from scipy import stats
from pyarrow import parquet as pq
from matplotlib import pyplot as plt
from itertools import combinations, permutations

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', None)

import time
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print ('%r  %2.2f ms' %(method.__name__, (te - ts) * 1000))
        return result
    return timed

#multi-level distributions calculation
stat_dict={'cnt':'count',
           'min':np.min,'mean':np.mean,               
           'prct_5' : lambda x: np.percentile(x,5),
           'prct_25': lambda x: np.percentile(x,25),
           'prct_40': lambda x: np.percentile(x,40),
           'prct_50': lambda x: np.percentile(x,50),
           'prct_60': lambda x: np.percentile(x,60),
           'prct_75': lambda x: np.percentile(x,75),
           'prct_95': lambda x: np.percentile(x,95),           
            'max':np.max} 
def dist_fun(df,var,level): 
    msk=df[var].notnull()
    return df.loc[msk].groupby(level,as_index=False)[var].agg(stat_dict,axis=1)

def mrg_dist(df,level):
    per_unit=dist_fun(df,"allowedAmountPerUnit",level)
    per_quan=dist_fun(df,"allowedAmountPerQuantity",level)
    total=dist_fun(df,"allowedAmount",level)
    mrg=total.merge(per_unit,on=level,suffixes=["_all","_unit"])
    mrg=mrg.merge(per_quan,on=level,suffixes=["","_quan"])
    return mrg.reset_index(level=0)

@timeit
def multi_lv_dist(df,main,sub):
    dist=pd.DataFrame() 
    dist=mrg_dist(df,level=main) 
    dist["level"]=["_".join(main) for i in range(len(dist))]
    for lv_cnt in range(1,len(sub)+1):
            for lv in combinations(sub,lv_cnt):
                d=mrg_dist(df,level=main+list(lv))
                d["level"]=["_".join(main+list(lv)) for i in range(len(d))]
                dist=dist.append(d)             
    return dist

    