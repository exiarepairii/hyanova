# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm

def get_all_true_subsets(s):
    k = len(s)
    tmp = []
    subsets = []
    for i in range(k):
        tmp.extend(list(combinations(s,i)))
    for t in tmp:
        subsets.append(list(t))
    return subsets

def get_a_u(cal_df,u,ne_u):
    a_u = cal_df.groupby(list(u)).mean().reset_index()
    if len(ne_u):
        for nu in ne_u:
            a_u[nu] = np.nan
    
    col_name=cal_df.columns.tolist()
    a_u = a_u.reindex(columns=col_name)
    return a_u
    
def get_f_w(f,u):
    sum_f_w = pd.DataFrame(columns=f.columns[:])
    w_list = get_all_true_subsets(u)
    for w in w_list:
        sum_f_w = sum_f_w.append(f[f['id']==hash(str(w))], ignore_index=True)
        
    return sum_f_w


def get_f_u(a_u,sum_f_w,u):
    col_name=sum_f_w.columns.tolist()
    a_u = a_u.reindex(columns=col_name)
    for row_index,r in sum_f_w.iterrows():
        r2 = r[1:-1]
        not_null_index = r2.notnull().values
        if not not_null_index.any(): # r全为nan
            a_u.loc[:,col_name[-1]] -= r[-1]
        else:
            equal_index = (a_u.iloc[:,1:-1]==r2[not_null_index]).values.sum(axis=1)
            a_u.loc[equal_index==not_null_index.sum(),col_name[-1]] -= r[-1]
    a_u['id'] = hash(str(list(u)))
    return a_u

def fanova(cal_df,params,measure):
    # cal_df.columns = [params,measure]
    axis = ['id']
    axis.extend(params)
    axis.append(measure)
    f = pd.DataFrame(columns=axis)
    f.loc[0] = np.nan
    f[measure] = cal_df[measure].mean()
    f['id'] = hash(str([]))
    v_all = np.std(cal_df[measure].to_numpy())**2
    v = pd.DataFrame(columns=['u','v_u','F_u(v_u/v_all)'])
    with tqdm(total=2**len(params)-1) as pbar:
        for k in range(1, len(params) + 1):
            for u in combinations(params, k):
                # calculate a_u
                ne_u = set(params)-set(u)
                a_u = get_a_u(cal_df,u,ne_u)
                sum_f_w = get_f_w(f,u)
                f_u = get_f_u(a_u,sum_f_w,u)
                f = f.append(f_u,ignore_index=True)
                v = v.append({'u':u,'v_u':(f_u[measure]**2).mean(),'F_u(v_u/v_all)':(f_u[measure]**2).mean()/v_all},ignore_index=True)
                pbar.update(1)
                
    return v
    
def read_csv(path,measure):
    df = pd.read_csv(path).iloc[:,1:]
    params = list(eval(df.loc[0,'params']).keys())
    df = df[['params',measure]]
    for key in params:
        df[key] = df['params'].apply(lambda x:eval(x)[key])
    
    df.drop(['params'],axis=1,inplace=True)
    col_name = params[:]
    col_name.append(measure)
    df = df.reindex(columns=col_name)
    return df,params


def main():
    measure = 'mean_test_score'
    path = './iris[GridSearchCV]Model1.csv'
    df,params = read_csv(path,measure)
    importance = fanova(df,params,measure)
    importance.to_csv('importance.csv')
                
if __name__ == "__main__":
    main()