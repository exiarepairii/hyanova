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
        tmp.extend(list(combinations(s, i)))
    for t in tmp:
        subsets.append(list(t))
    return subsets


def get_a_u(cal_df, u, ne_u):
    a_u = cal_df.groupby(list(u)).mean().reset_index()
    if len(ne_u):
        for nu in ne_u:
            a_u.loc[:,nu] = np.nan

    col_name = cal_df.columns.tolist()
    a_u = a_u.reindex(columns=col_name)
    return a_u


def get_f_w(f, u):
    sum_f_w = pd.DataFrame(columns=f.columns[:])
    w_list = get_all_true_subsets(u)
    for w in w_list:
        sum_f_w = sum_f_w.append(f[f['id'] == hash(str(w))], ignore_index=True)

    return sum_f_w


def get_f_u(a_u, sum_f_w, u):
    col_name = sum_f_w.columns.tolist()
    a_u = a_u.reindex(columns=col_name)
    for row_index, r in sum_f_w.iterrows():
        r2 = r[1:-1]
        not_null_index = r2.notnull().values
        if not not_null_index.any():  # r全为nan
            a_u.loc[:, col_name[-1]] -= r[-1]
        else:
            equal_index = (a_u.iloc[:, 1:-1] ==
                           r2[not_null_index]).values.sum(axis=1)
            a_u.loc[equal_index == not_null_index.sum(), col_name[-1]] -= r[-1]
    a_u['id'] = hash(str(list(u)))
    return a_u


def analyze(cal_df):
    """
        Use `hyanova.analyze(df)` to do the functional ANOVA decomposition. 
        It needs a <pnadas.DataFrame> object, and will return a result in <pnadas.DataFrame> type
    """
    axis = ['id']
    axis.extend(list(cal_df.columns))
    params = axis[1:-1]
    metric = axis[-1]
    f = pd.DataFrame(columns=axis)
    f.loc[0,:] = np.nan
    f.loc[0,metric] = cal_df[metric].mean()
    f.loc[0,'id'] = hash(str([]))
    v_all = np.std(cal_df[metric].to_numpy())**2
    v = pd.DataFrame(columns=['u', 'v_u', 'F_u(v_u/v_all)'])
    with tqdm(total=2**len(params) - 1) as pbar:
        for k in range(1, len(params) + 1):
            for u in combinations(params, k):
                # calculate a_u
                ne_u = set(params) - set(u)
                a_u = get_a_u(cal_df, u, ne_u)
                sum_f_w = get_f_w(f, u)
                f_u = get_f_u(a_u, sum_f_w, u)
                f = f.append(f_u, ignore_index=True)
                tmp_f_u = f_u.loc[:,metric].to_numpy()
                v = v.append({'u': u, 'v_u': (tmp_f_u**2).mean(), 'F_u(v_u/v_all)': (
                    tmp_f_u**2).mean() / v_all}, ignore_index=True)
                pbar.update(1)

    return v


def read_csv(path, metric):
    """
    You can use `read_csv(path, metric)` to load data from a csv file. It works same as read_df(), and will return two objects.
        1. a `DataFrame` with all hyperparameters' value and the value of metric you choose
        2. a `list` of all hyperparameters' name
    """
    df = pd.read_csv(path).iloc[:, 1:]
    return read_df(df, metric)


def read_df(df, metric):
    """
    You can use `read_df(df,metric)` to load data from a `<class 'pandas.core.frame.DataFrame'>` object. It will return two objects.
        1. a `DataFrame` with all hyperparameters' value and the value of metric you choose
        2. a `list` of all hyperparameters' name
    """
    params = list(eval(df.loc[0, 'params']).keys())
    result = pd.DataFrame(df.loc[:,metric].copy())
    tmp_df = df.loc[:,'params'].copy()
    for key in params:
        result.insert(loc=0, column=key, value=tmp_df.apply(lambda x: eval(x)[key]))

    col_name = params[:]
    col_name.append(metric)
    result = result.reindex(columns=col_name).copy()
    return result, params


def test():
    metric = 'mean_test_score'
    path = './iris[GridSearchCV]Model1.csv'
    df, params = read_csv(path, metric)
    importance = analyze(df)
    importance.to_csv('importance.csv')

if __name__ == "__main__":
    test()
