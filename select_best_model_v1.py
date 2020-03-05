# -*- coding: utf-8 -*-
'''
Created on Fri Jan 17 14:11:36 2020

@author: wakabayashi
filename: pred_based_salesSchedule
'''
import sys
import os
import copy
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates
#import math
import pandas as pd
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import warnings
import matplotlib.pyplot as plt
import shutil
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler #正規化
from sklearn.metrics import r2_score
import statsmodels.api as sm #トレンド + 季節成分に分解
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV #ridge回帰
from sklearn.linear_model import LassoLarsCV #lasso回帰
from sklearn.linear_model import LassoLarsIC #lasso回帰
from sklearn.linear_model import ElasticNetCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as RFR #random forest
from xgboost import XGBRegressor #xgboost

### 変数選択 ###
from tsfresh import extract_features #特徴抽出
import shap
from boruta import BorutaPy

'''
# 自作モジュール追加
myfunc_path = 'D:/Work/TAC/20200116'
sys.path.append(myfunc_path)

%matplotlib qt
%matplotlib inline
'''

#%% import my function
from common import calc_mean_error_rate
from common import conv_timestamp
from common import get_overlap_item_idxs
from common import make_no_overlap_data
#from common import make_subplot
#from common import get_comb_key
#from common import make_set_item_data
from common import select_output_var
from common import var_select_based_contr_rate
from common import extract_car_data
from common import conv_comb2matrix
#from common import ide_columns_from_tbl
#from common import reshape_carSales_data
from common import split_list
from common import get_file_list
from common import rmse

### モデル構築 ###
from common import train_test_ridge
from common import train_test_lasso
from common import train_test_en
from common import train_test_randomForest
from common import train_test_xgb

### 結果出力 ###
from common import output_result_detail

### 特徴抽出 ###
from common import gen_lag_data
#from common import get_1year_before_data
from common import calc_stats_in_window

### グラフ表示 ###
#from common import make_subplot

plt.rcParams['font.family'] = 'IPAexGothic'
warnings.simplefilter('ignore')


# %% 
def plot_time_series(fname, t, y, y_pred, ref_data, yabel_name, title_name, xlimit=None, interval=1, fontsize=14, figsize=(12,7),  save_flag=1):
    '''
    時系列プロット
    実績、内示、予測の3種類の時系列をプロット
    '''
    tmp = np.concatenate([y, ref_data, y_pred], 0)
    ylimit = [0, max(tmp)]
    #ylimit = [min(tmp), max(tmp)]
    #ylimit = [0.5*min(tmp), max(tmp)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(t, y, 'ko-') #実績
    ax.plot(t, ref_data, 'bo-') #内示
    ax.plot(t, y_pred, 'ro-') #予測
    
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval, tz=None))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
    
    plt.setp(ax.get_xticklabels(), rotation=90) #x軸目盛回転
    
    ax.grid()
    
    if xlimit != None:
        plt.xlim(xlimit)
    plt.ylim(ylimit)
    plt.rcParams['font.size'] = fontsize
    plt.ylabel(yabel_name)
    plt.title(title_name)
    
    # グラフ保存
    if save_flag:
        plt.savefig(fname)

#%%
def replace_forecast_in_out_of_range(y, y_pred, forecast, valid_key, test_key, horizon=2, w=4):
    '''データ範囲を超える場合、内示で置換
    '''
    ### データ範囲の算出 ###
    length = len(y) #データ長
    limit = np.zeros([length,2]) #記録領域
    
    for k in range(horizon+w,length):
        end = k-horizon
        start = end - w
        
        index = np.arange(start, end)
        index = np.intersect1d(index, valid_key)
        tmp = y.iloc[index]
        
        limit[k,0] = tmp.min()
        limit[k,1] = tmp.max()
    
    ### 範囲を超える場合、内示で置換 ###
    bool1 = y_pred < limit[test_key,0]
    bool2 = limit[test_key,1] < y_pred
    bools = bool1 + bool2
    
    index = np.where(bools == 1)[0]
    
    prediction = copy.deepcopy(y_pred)
    prediction[index] = forecast.iloc[test_key[index]]
    return prediction, index

#%%
def extract_car_data2(car_data_o, item_tbl):
    '''指定した車種分類1/2を持つデータを抽出
    '''
    car_type1 = list(np.unique(item_tbl['type1']))[0]
    car_type2 = list(np.unique(item_tbl['type2']))
    
    index1 = np.where(car_type1 == car_data_o['type1'])[0]
    index1 = np.sort(index1)
    
    index2 = []
    for i in range(len(car_type2)):
        idx = np.where(car_type2[i] == np.array(car_data_o['type2'].values))[0]
        index2 = index2 + list(idx)
    index2 = np.sort(index2)
    
    index = np.intersect1d(index1,index2)
    
    x = car_data_o.iloc[index,:]
    return x

#%%
def reshape_car_product_data(output_dir, x, patterns, lags):
    '''生産計画データ整形
    '''
    for i in range(len(lags)):
        lag = lags[i]
        col_name = patterns[i] + '月生産計画'
        #dir_name = copy.deepcopy(col_name)
        columns = ['車種分類1','車種分類2',col_name]
        buffer = x[columns]
        buffer = buffer.rename(columns={col_name:'value'})
        
        x2 = conv_comb2matrix(buffer) #myfun
        columns = list(x2.columns)
        for j in range(len(columns)):
            columns[j] = columns[j] + '_' + patterns[i]
            x2.columns = columns
        
        t = x2.index
        t = [t[k] + relativedelta(months=lag) for k in range(len(t))]
        x2.index = t
        
        if i >= 1:
            car_data = pd.concat([car_data,x2], axis=1) #結合
        else:
            car_data = copy.deepcopy(x2)
    
    ### save ###
    car_name = car_data.columns[0]
    fname = os.path.join(output_dir, car_name+'.csv')
    car_data.to_csv(fname, encoding='cp932')
    return car_data

#%%
def list_to_dataframe(comb_lists, columns):
    '''レコード情報を生成
    '''
    for i in range(len(comb_lists)):
        buffer = pd.Series(comb_lists[i], index=columns)
        buffer = pd.DataFrame(buffer).T
        
        if i >= 1:
            comb_tbl = pd.concat([comb_tbl, buffer])
        else:
            comb_tbl = copy.deepcopy(buffer)
    
    comb_tbl.index = np.arange(comb_tbl.shape[0]) # index振り直し
    return comb_tbl

#%% 
def get_comb_key(data):
    '''組合せキー取得
    '''
    columns = list(data.columns)
    
    keys = list(map(list, set(map(tuple, data.values))))
    keys = np.array(keys) #配列に変換
    
    # 多重キーによるソート
    keys = sorted(keys, key=lambda x:(x[0],x[1],x[2],x[3]), reverse=False)
    keys = np.array(keys) #配列に変換
    
    keys = pd.DataFrame(keys, columns=columns) #データフレームに変換
    return keys

#%%
def conv_comb2matrix_2(x, comb_tbl):
    '''各条件をカラム列に持つテーブルデータの生成
    '''
    columns = list(x.columns)
    columns = columns[0:-1]
    
    for i in range(comb_tbl.shape[0]):
        for col_id in range(len(columns)):
            col_name_ = columns[col_id]
            string = comb_tbl[col_name_].iloc[i] #条件
            
            idx = np.where(string == x[col_name_])[0]
            if col_id >= 1:
                index1 = np.intersect1d(index1, idx)
                col_name = col_name + '_' + string
            else:
                index1 = copy.deepcopy(idx)
                col_name = copy.deepcopy(string)
        
        buffer = x[h_name].iloc[index1]
        buffer.name = col_name
        
        if i >= 1:
            car_data = pd.concat([car_data,buffer], axis=1)
        else:
            car_data = copy.deepcopy(buffer)
    
    # 値が1つも存在しない列を削除
    buffer = car_data.fillna(0)
    index = np.where(buffer.sum() != 0)[0]
    
    car_data = car_data.iloc[:,index] #車両生産計画
    return car_data

#%%
def gen_cond_dicts(conds):
    '''検索条件の生成
    '''
    columns = ['仕様書','車種','トリム','色']
    conds = conds[columns]
    
    
    # 辞書型に変換
    columns = list(conds.index)
    for i in range(len(columns)):
        buffer = conds.iloc[i]
        
        # 文字列内にカンマ区切りが存在しない場合
        if buffer.find(',') != -1:
            buffer = list(map(str, buffer.split(',')))
        else:
            buffer = copy.deepcopy([buffer])
        
        if i >= 1:
            tmp.update({columns[i]: buffer})
        else:
            tmp = {columns[i]: buffer}
    conds = copy.deepcopy(tmp) #結果代入
    return conds

#%%
def extract_data_in_fulfill_condition(car_data_o, conds):
    '''条件を満たすデータ切り出し
    '''
    columns = list(conds.keys())
    
    index2 = []
    for i in range(len(columns)):
        col_name = columns[i]
        conds_ = conds[col_name]
        
        # 条件が存在しない場合⇒スキップ
        if conds_[0] == '':
            continue
        
        for j in range(len(conds_)):
            idx = np.where(conds[col_name][j] == car_data_o[col_name].values)[0]
            if j >= 1:
                index1 = np.append(index1, idx)
            else:
                index1 = copy.deepcopy(idx)
        
        if i >= 1:
            index2 = np.intersect1d(index2, index1)
        else:
            index2 = copy.deepcopy(index1)
    
    x = car_data_o.iloc[index2,:]
    return x

#%%
def agg_data(x, columns):
    '''集約
    '''
    cols = copy.deepcopy(columns)
    cols.append('年月')
    x2 = x.groupby(cols).sum()
    
    
    # レコード情報を生成
    comb_tbl = list_to_dataframe(list(x2.index), cols) #myfun
    
    x2.index = np.arange(x2.shape[0]) #index振り直し
    x2 = pd.concat([comb_tbl, x2], axis=1) #レコード情報を結合
    
    x2.index = x2['年月'] #indexにタイムスタンプを入れる
    x2 = x2.drop(columns=['年月']) #不要列の削除
    return x2

#%%
def get_carLists_include_pattern(item_tbl, car_data):
    '''
    品番紐付表の車種列を修正
    文字列を含む全車種名を取得
    '''
    new_item_tbl = copy.deepcopy(item_tbl)
    
    N = item_tbl.shape[0]
    for i in range(N):
        if i%50 == 0:
            print('NO.', str(i), '/', str(N-1))
        car_type = item_tbl['車種'].iloc[i]
        
        # 例外処理
        if car_type == '':
            continue
        
        index = np.where(car_data['車種'].str.contains(car_type) == 1)[0]
        car_list = list(car_data['車種'].iloc[index].unique())
        
        for j in range(len(car_list)):
            if j >= 1:
                string = string + ',' + car_list[j]
            else:
                string = car_list[j]
        
        new_item_tbl['車種'].iloc[i] = string
    return new_item_tbl

#%%
def reshape_item_tbl(item_tbl, car_data):
    '''除外パターンを削除し、それ以外のパターンを要素に入れる
    '''
    lists = np.where(item_tbl['except_label'] != '')[0]
    
    for i in range(len(lists)):
        conds = copy.deepcopy(item_tbl.iloc[lists[i]])
        
        ### 検索条件の生成 ###
        columns = ['仕様書','車種','トリム','色']
        conds_dict = gen_cond_dicts(conds[columns]) #myfun
        
        
        # セルに記載されたカラム列の名称を取得
        col_name = conds['except_label']
        
        # 指定した要素の削除
        del_patterns = conds_dict[col_name]
        conds_dict[col_name] = ['']
        
        
        # 条件を満たすデータ切り出し
        x = extract_data_in_fulfill_condition(car_data, conds_dict) #myfun
        
        # レコード情報を生成
        comb_tbl = x.groupby(columns).sum()
        comb_tbl = list_to_dataframe(list(comb_tbl.index), columns) #myfun
        
        # 削除パターンの削除
        index = []
        for j in range(len(del_patterns)):
            idx = np.where(comb_tbl[col_name] != del_patterns[j])[0]
            index = index + list(idx)
        index = np.sort(np.unique(index))
        comb_tbl = comb_tbl.iloc[index,:]
        
        # 残った要素を紐付表に入れる
        include_patterns = list(comb_tbl[col_name].unique())
        
        # listを文字列に変換
        for j in range(len(include_patterns)):
            if j >= 1:
                buffer = buffer + ',' + include_patterns[j]
            else:
                buffer = include_patterns[j]
        include_patterns = copy.deepcopy(buffer)
        
        conds[col_name] = include_patterns #結果代入
        
        item_tbl.iloc[lists[i]] = conds #結果代入
    return item_tbl

#%%
def make_subplot(fname, t, y, title_name, fontsize=14, figsize=(12,10), save_flag=1):
    '''subplotの描画
    '''
    xlimit = [t.min(), t.max()]
    
    dim = y.shape[1]
    
    fig = plt.figure(figsize=figsize)
    
    for i in range(dim):
        ylabel_name = y.columns[i]
        
        ax = fig.add_subplot(dim,1,i+1)
        ax.plot(t, y.iloc[:,i].values, 'bo-')
        
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2, tz=None))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
        
        plt.setp(ax.get_xticklabels(), rotation=90) #x軸目盛回転
        
        plt.xlim(xlimit)
        plt.grid()
        plt.rcParams['font.size'] = fontsize
        plt.ylabel(ylabel_name)
        
        if i == 0:
            plt.title(title_name)
    
    # グラフ保存
    if save_flag:
        plt.savefig(fname)

#%%
def feature_select_based_r2(x, y, n_cv=5):
    '''決定係数による変数選択
    '''
    valid_key = np.where(y.values != 0)[0] #有効なキーの取得
    
    # 分割後のデータのインデックス
    segment_index = list(split_list(valid_key, round(len(valid_key)/n_cv)))
    
    dim = x.shape[1]
    scores = pd.Series(np.zeros(dim), index=x.columns) #記録領域
    
    # 説明変数の指定
    for i in range(dim):
        y_pred = np.zeros(len(y)) #記録領域
        
        # 分割したデータのインデックス指定
        for seg_id in range(len(segment_index)):
            test_key = segment_index[seg_id]
            train_key = np.array(list(set(valid_key) - set(test_key)))
            
            x_train = x.iloc[train_key,i].values.reshape(-1,1)
            y_train = y.iloc[train_key].values.reshape(-1,1)
            x_test = x.iloc[test_key,i].values.reshape(-1,1)
            
            
            ### 正規化 ###
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            
            x_scaler.fit(x_train)
            y_scaler.fit(y_train)
            
            x_train = x_scaler.transform(x_train)
            y_train = y_scaler.transform(y_train)
            x_test = x_scaler.transform(x_test)
            
            
            ### train/test ###
            clf = LinearRegression().fit(x_train, y_train) #学習
            ym = clf.predict(x_test) #予測
            
            y_pred[test_key] = y_scaler.inverse_transform(ym).reshape(-1) #非正規化
        
        scores.iloc[i] = r2_score(y.iloc[valid_key].values, y_pred[valid_key]) #決定係数
    return scores, y_pred

#%%
def select_train_data(y, alpha=3):
    '''
    学習範囲の選択
    直近のデータに対して一定の範囲内に入るデータを学習データとして選択する
    '''
    mu = y.iloc[-1] #平均
    sigma = y.std() #標準偏差
    limit = np.zeros(2) #記録領域
    limit[0] = mu - alpha*sigma/2 #下限
    limit[1] = mu + alpha*sigma/2 #上限
    
    idx1 = np.where(limit[0] <= y)[0]
    idx2 = np.where(y <= limit[1])[0]
    train_key = np.intersect1d(idx1,idx2)
    
    # 例外処理
    error_flag = 0 #初期化
    if len(train_key) <= 3:
        train_key = np.arange(len(y))
        error_flag = 1
    return train_key, error_flag

#%%
#def get_1year_before_data(data):
#    '''同月の1年前のデータを説明変数に追加
#    '''
#    ### 連続したタイムスタンプとなるように、データ整形 ###
#    start = data.index[0]
#    end = data.index[-1]
#    t = pd.date_range(start, end, freq='MS')
#    t = pd.Series(np.zeros(len(t)), index=t)
#    x = pd.concat([t, data], axis=1)
#    x = x.iloc[:,1]
#    
#    ### 月を取得 ###
#    t = data.index
#    months = [t[k].month for k in range(len(t))]
#    months = np.array(months)
#    
#    n = len(months) #データ長
#    
#    ### 同月1年前の統計量 ###
#    #columns = ['同月1年前', '同月1年前_mean', '同月1年前_std', '同月1年前_max']
#    columns = ['同月1年前', '同月1年前_mean', '同月1年前_std','同月過去_mean','同月過去_std']
#    x = pd.DataFrame(np.zeros([n,len(columns)]), columns=columns, index=data.index) #記録領域
#    for k in range(n):
#        index = np.where(months[k] == months[0:k])[0]
#        if len(index) != 0:
#            index = index[-1]
#            x['同月1年前'].iloc[k] = data.iloc[index]
#            x['同月1年前_mean'].iloc[k] = data.iloc[index-1:index+2].mean()
#            x['同月1年前_std'].iloc[k] = data.iloc[index-1:index+2].std()
#            #x['同月1年前_max'].iloc[k] = data.iloc[index-1:index+2].max()
#    
#    
#    ### 同月過去の統計量 ###
#    for k in range(n):
#        index = np.where(months[k] == months[0:k])[0]
#        if len(index) != 0:
#            x['同月過去_mean'].iloc[k] = data.iloc[index].mean()
#            #x['同月過去_mean'].iloc[k] = data.iloc[index-1:index+2].mean()
#            x['同月過去_std'].iloc[k] = data.iloc[index].std()
#    return x

def get_1year_before_data(data):
    '''同月の1年前のデータを説明変数に追加
    '''
    ### 連続したタイムスタンプとなるように、データ整形 ###
    start = data.index[0]
    end = data.index[-1]
    t = pd.date_range(start, end, freq='MS')
    t = pd.Series(np.zeros(len(t)), index=t)
    x = pd.concat([t, data], axis=1)
    x = x.iloc[:,1]
    
    ### 月を取得 ###
    t = data.index
    months = [t[k].month for k in range(len(t))]
    months = np.array(months)
    
    n = len(months) #データ長
    
    ### 同月1年前の統計量 ###
    #columns = ['同月1年前', '同月1年前_mean', '同月1年前_std', '同月1年前_max']
    columns = ['同月1年前', '同月1年前_mean', '同月1年前_std','同月過去_mean','同月過去_std']
    if 0:
        length = len(columns)
        w = 3
        for lag in range(w):
            columns.append('同月1年前_lag'+str(1+lag))
    
    x = pd.DataFrame(np.zeros([n,len(columns)]), columns=columns, index=data.index) #記録領域
    for k in range(n):
        index = np.where(months[k] == months[0:k])[0]
        if len(index) != 0:
            index = index[-1]
            x['同月1年前'].iloc[k] = data.iloc[index]
            x['同月1年前_mean'].iloc[k] = data.iloc[index-1:index+2].mean()
            x['同月1年前_std'].iloc[k] = data.iloc[index-1:index+2].std()
            #import pdb; pdb.set_trace()
            ### 同月1年前のラグ変数 ###
            if 0:
                if index-w >= 0:
                    buffer = data.iloc[index-w:index].values
                    buffer = np.flip(buffer)
                    x.iloc[k,length:x.shape[1]] = buffer
    
    
    ### 同月過去の統計量 ###
    for k in range(n):
        index = np.where(months[k] == months[0:k])[0]
        if len(index) != 0:
            x['同月過去_mean'].iloc[k] = data.iloc[index].mean()
            #x['同月過去_mean'].iloc[k] = data.iloc[index-1:index+2].mean()
            x['同月過去_std'].iloc[k] = data.iloc[index].std()
    return x

#%%
def load_monthly_trend(dir_path):
    '''月別トレンドの取得
    '''
    # ファイル一覧の取得
    FileNames = get_file_list(dir_path) #myfun
    
    item_names = []
    for i in range(len(FileNames)):
        fname = FileNames[i]
        _, fname = fname.split('_')
        buffer = fname.replace('.csv','')
        item_names.append(buffer)
    
    index = np.where(output_name == np.array(item_names))[0][0]
    
    fname = FileNames[index]
    
    ### データ読み込み ###
    filepath = os.path.join(dir_path, fname)
    monthly_trend = pd.read_csv(filepath, parse_dates=True, header=0, index_col=0, encoding='cp932')
    return monthly_trend

#%%
def normalize(x, train_key):
    '''正規化
    '''
    scaler = StandardScaler()
    
    if x.ndim == 1:
        x_train = x[train_key].reshape(-1,1)
        scaler.fit(x_train)
        x_norm = scaler.transform(x.reshape(-1,1)).reshape(-1)
    else:
        x_train = x[train_key,:]
        scaler.fit(x_train)
        x_norm = scaler.transform(x)
    return x_norm, scaler

#%%
#def output_result_detail(fname, actual, forecast, prediction, test_key):
#    """結果明細の出力
#    """
#    mdl_lists = list(prediction.columns)
#    
#    #columns = ['timestamp', 'test_flag', '実績', 'N+2月内示', '予測', '内示誤差率 [%]', '予測誤差率 [%]']
#    columns = ['timestamp', 'test_flag', '実績', 'N+2月内示']
#    columns = columns + mdl_lists
#    columns.append('内示誤差率 [%]')
#    for i in range(len(mdl_lists)):
#        columns.append(mdl_lists[i]+'_誤差率 [%]')
#    
#    n = len(actual) #データ長
#    res = pd.DataFrame(np.zeros([n,len(columns)]), columns=columns, index=np.arange(1,n+1)) #記録領域
#    
#    res['timestamp'] = actual.index
#    res['test_flag'].iloc[test_key] = 1
#    
#    res['実績'] = actual.values
#    res['N+2月内示'] = forecast.values
#    
#    # 予測データの整形
#    dim = prediction.shape[1]
#    tmp = np.zeros([n,dim])
#    tmp[test_key,:] = prediction
#    res[mdl_lists] = tmp
#    
#    # 誤差率の計算
#    res['内示誤差率 [%]'] = abs((actual.values - forecast.values) / actual.values) * 100
#    #res['予測誤差率 [%]'].iloc[test_key] = abs((actual.iloc[test_key].values - res['予測'].iloc[test_key].values) / actual.iloc[test_key].values) * 100
#    for i in range(len(mdl_lists)):
#        col_name = mdl_lists[i]
#        res[col_name+'_誤差率 [%]'].iloc[test_key] = abs((actual.iloc[test_key].values - res[col_name].iloc[test_key].values) / actual.iloc[test_key].values) * 100
#    
#    
#    ### save ###
#    res.to_csv(fname, encoding='cp932')
#    
#    return res

###############################################################################
#%% initialize
root_dir = 'D:/Work/TAC/20200116'
base_dir = os.path.join(root_dir, 'ForAllData')
data_dir = os.path.join(root_dir, 'data')
input_dir = os.path.join(root_dir, 'input')
#output_dir = os.path.join(base_dir, 'output/test')
output_dir = os.path.join(base_dir, 'output/20200305_最適モデル選択/ver01')
dir_path = os.path.join(output_dir, 'fig')
if os.path.isdir(dir_path) == False:
    os.makedirs(dir_path)

# データ読込 : 納入実績
fname = os.path.join(data_dir, '納入実績.csv')
output_data_o = pd.read_csv(fname, parse_dates=True, header=0, index_col=None, encoding='cp932')

# データ読込 : 内示
fname = os.path.join(data_dir, 'forecast.csv')
forecast_o = pd.read_csv(fname, parse_dates=True, header=0, index_col=0, encoding='cp932')

# データ読込 : 車両生産計画
#fname = os.path.join(data_dir, '車両生産計画_集約.csv')
fname = os.path.join(data_dir, '車両生産計画.csv')
car_data_o = pd.read_csv(fname, parse_dates=True, header=0, index_col=0, encoding='cp932')

# データ読込 : 車種-品番紐づけ表
#fname = os.path.join(input_dir, 'マット品番リスト_生産計画対応表.csv')
fname = os.path.join(input_dir, 'マット品番リスト3.csv')
#item_tbl_o = pd.read_csv(fname, header=0, index_col=0)
item_tbl_o = pd.read_csv(fname, header=0, index_col=None, encoding='cp932')

# タイムスタンプの型変換
col_name = '年月'
forecast_o[col_name] = conv_timestamp(forecast_o[col_name], date_format='%Y-%m-%d') #myfun

#%% スクリプトのコピー
dir_path, fname = os.path.split(__file__)
output_file = os.path.join(output_dir, fname)
shutil.copyfile(__file__, output_file)

#%% 前処理(目的変数)
# タイムスタンプの型変換
start_idx = np.where('規格' == output_data_o.columns)[0][0] + 1

n_col = output_data_o.shape[1]
t = output_data_o.columns[start_idx:n_col]
t = conv_timestamp(t, date_format='%Y/%m/%d') #myfun

columns = list(output_data_o.columns)
columns[start_idx:n_col] = t
output_data_o.columns = columns

# 重複のある商品コードをリストアップ
comb_indexs = get_overlap_item_idxs(output_data_o['商品コード'].values) #myfun

# 商品コードに重複のない出荷実績データを生成
output_data_o = make_no_overlap_data(output_data_o, comb_indexs) #myfun
output_data_o['規格'] = output_data_o['規格'].fillna('')


#%% 前処理(車両生産計画)
car_data = copy.deepcopy(car_data_o)

car_data = car_data.drop(columns=['ライン'])

# 車種分類1列の540A⇒341Bに置換
patterns = ['540A','341B']
col_name = '車種分類1'
car_data[col_name] = car_data[col_name].replace(patterns[0],patterns[1])

# タイムスタンプ型変換
col_name = '年月'
car_data[col_name] = conv_timestamp(car_data[col_name], '%Y年%m月') #myfun

# nanを置換
col_name = '車種分類2'
car_data[col_name] = car_data[col_name].fillna('')

car_data_o = copy.deepcopy(car_data)
item_tbl = copy.deepcopy(item_tbl_o)


# 不要列の削除
columns = ['N月輸出', 'N+1月輸出', 'N+2月輸出']
car_data_o = car_data_o.drop(columns=columns)

# カラム名の置換
columns = list(car_data_o.columns)
for i in range(len(columns)):
    columns[i] = columns[i].replace('市販','生産計画')
car_data_o.columns = columns

# 指定した列のデータ型を変換
col_name = '色'
car_data_o[col_name] = car_data_o[col_name].astype('str')


#%% 前処理(紐付表)
# NANの要素を空白に置換
item_tbl = item_tbl.fillna('')

# set output filename
fname = os.path.join(output_dir, 'マット品番リスト.csv')

if 0:
    '''
    品番紐付表の車種列を修正
    文字列を含む全車種名を取得
    '''
    item_tbl = get_carLists_include_pattern(item_tbl, car_data_o) #myfun
    item_tbl = item_tbl.rename(columns={'Unnamed: 0': ''})
    
    ### 除外パターンを削除し、それ以外のパターンを要素に入れる ###
    item_tbl = reshape_item_tbl(item_tbl, car_data_o) #myfun
    
    # save
    item_tbl.to_csv(fname, index=None, encoding='cp932')
# load
item_tbl = pd.read_csv(fname, header=0, index_col=False, encoding='cp932')

# NANの要素を空白に置換
item_tbl = item_tbl.fillna('')


### 品番名リストの取得 ###
item_codes = list(np.unique(item_tbl['品番']))
item_codes.remove('')

N = len(item_codes) #品番数

index = []
for i in range(N):
    if item_codes[i] in output_data_o['商品コード'].values:
        index.append(i)

item_codes = [item_codes[index[i]] for i in range(len(index))]


#%% 目的変数の選択
threshold = 20

data = output_data_o.iloc[:, start_idx:output_data_o.shape[1]]

# 指定した条件を満たす品目を抽出
index, stats = select_output_var(data, output_data_o['商品コード'], threshold) #myfun

# 対象品目リストと、目的変数リストの両方に存在する品目IDを取得
tmp = list(output_data_o['商品コード'].iloc[index])
item_lists = []
for i in range(len(item_codes)):
    if item_codes[i] in tmp:
        item_lists.append(i)

N = len(item_codes) #全品目数

#### 平均誤差率の記録領域 ###
#mdl_lists = ['ridge','en','rf','xgb','svr']
#mdl_lists = ['ridge','en','rf','xgb']

index = []
for i in range(N):
    idx = np.where(item_codes[i] == output_data_o['商品コード'].values)[0][0]
    index.append(idx)

columns = ['内示誤差率 [%]', '予測誤差率 [%]']
all_err_rate = output_data_o.iloc[index, 0:start_idx]
all_err_rate.index = np.arange(N) #index振り直し

for i in range(len(columns)):
    col_name = columns[i]
    all_err_rate[col_name] = np.zeros(N)


#%% 説明/目的変数の生成
#t_start_dict = {18:'2016/10/1',
#                20:'2017/12/1',
#                39:'2017/8/1',
#                40:'2017/6/1',
#                49:'2017/1/1'}
#t_start_dict = {'0821028U50C0':'2017/1/1', #63
#                '0821028W00C0':'2017/5/1'} #81

t_start_str = '2017/3/1'
t_start = dt.datetime.strptime(t_start_str, '%Y/%m/%d')

N = len(item_lists)
start = 2
lists = np.arange(start,start+1)
lists = np.arange(92,N)
#lists = np.arange(N)

if 0:
    output_name = '0821026E60B0'
    output_id = np.where(output_name == np.array(item_codes))[0]
    output_id = int(output_id)
    
    output_id_ = np.where(output_id == np.array(item_lists))[0]
    output_id_ = int(output_id_)

# 目的変数の指定
for output_id_ in lists:
    #%%
    output_id = item_lists[output_id_]
    output_name = item_codes[output_id]
    
    i = np.where(output_name == output_data_o['商品コード'].values)[0][0]
    
    output_data = output_data_o.iloc[[i], start_idx:output_data_o.shape[1]]
    output_data = output_data.iloc[0,:]
    output_data.name = output_name
    
    date = '{0:%H:%M:%S}'.format(dt.datetime.now())
    print('[', date, '] START -> No.', output_id_, '/', len(item_codes), ' [No.', str(output_id), ']')
    
    
    ### 内示の取得 ###
    index = np.where(output_name == forecast_o['品番'].values)[0]
    forecast = forecast_o.iloc[index,:][['N+1月', 'N+2月']]
    
    # タイムスタンプの取得
    t = forecast_o.iloc[index,:]['年月']
    t = pd.to_datetime(t)
    
    forecast.index = t #indexをタイムスタンプに設定
    
    # 目的変数の形状に合うように、内示をリサイズ
    data = pd.concat([output_data, forecast], axis=1, join_axes=[output_data.index])
    forecast = data.iloc[:,1:data.shape[1]]
    
    # カラム名の変更
    forecast.columns = [forecast.columns[i] + '内示' for i in range(forecast.shape[1])]
    
    
    #%% グラフ表示(実績/予測の時系列 / 全期間)
    save_flag = 1
    fontsize = 17
    figsize = (16,10)
    
    # make output directory
    output_dir_ = os.path.join(output_dir, 'raw_all')
    if os.path.isdir(output_dir_) == False:
        os.makedirs(output_dir_)
    
    # set output filename
    fname = 'NO' + str(output_id) + '_' + output_name + '.png'
    fname = os.path.join(output_dir_, fname)
    
    y = copy.deepcopy(output_data)
    t = y.index
    
    n = len(t) #データ長
    tmp = np.zeros(n)
    tmp[tmp == 0] = np.nan
    plot_time_series(fname, t, y.values, tmp, forecast['N+2月内示'].values, output_name, '', None, 2, fontsize, figsize, save_flag)
    plt.close()
    
    
    #%% 車両生産計画データの加工
    ### 検索条件の生成 ###
    index = np.where(output_name == item_tbl['品番'])[0]
    index = int(index)
    conds = item_tbl.iloc[index,:]
    
    columns = ['仕様書','車種','トリム','色']
    conds = conds[columns]
    
    conds = gen_cond_dicts(conds) #myfun
    
    ### 条件を満たすデータ切り出し ###
    x = extract_data_in_fulfill_condition(car_data_o, conds) #myfun
    
    # 例外処理
    if x.shape[0] == 0:
        print('ERROR!')
        continue
    
    ### 集約 ###
    x2 = agg_data(x, columns) #myfun
    
    
    ### 合計値の算出 ###
    buffer1 = x2.groupby('年月').sum()
    cols = list(x2.columns)
    n = buffer1.shape[0]
    buffer2 = pd.DataFrame(np.zeros([n,len(cols)]), columns=cols, index=buffer1.index) #記録領域
    cols = buffer1.columns
    buffer2[cols] = buffer1
    
    buffer2[columns] = '合計'
    
    x2 = pd.concat([x2, buffer2]) #結合
    
    
    #%%
    ### 各条件をカラム列に持つテーブルデータの生成 ###
    # 条件一覧の取得(ヘッダ情報)
    comb_tbl = get_comb_key(x2[columns]) #myfun
    
    patterns = ['N+2']
    lags = [0]
    patterns = ['N+1','N+2']
    lags = [1,0]
    #patterns = ['N','N+1','N+2']
    #lags = [2,1,0]
    
    for h in range(len(patterns)):
        h_name = patterns[h] + '月生産計画'
        
        cols = copy.deepcopy(columns)
        cols.append(h_name)
        
        x3 = conv_comb2matrix_2(x2[cols], comb_tbl) #myfun
        
        cols = x3.columns
        cols = [cols[i] + '_' + patterns[h] for i in range(len(cols))]
        x3.columns = cols
        
        # タイムスタンプ遷移
        t = x3.index
        t = [t[k] + relativedelta(months=lags[h]) for k in range(len(t))]
        x3.index = t
        
        if h >= 1:
            car_data = pd.concat([car_data,x3], axis=1) #結合
        else:
            car_data = copy.deepcopy(x3)
    
    
    #%% 特徴抽出
    input_data = copy.deepcopy(car_data)
    input_labels = list(input_data.columns)
    
    #################
    # 内示の追加
    #################
    #--- N+1月内示
    if 1:
        x = forecast['N+1月内示']
        t = forecast.index
        x.index = t
        x.index = [t[k] + relativedelta(months=1) for k in range(len(t))] #タイムスタンプ遷移
        
        input_data = pd.concat([input_data, x], axis=1)
        input_labels.append(x.name) #説明変数のカラム名の追加
    
    #--- N+2月内示
    x = forecast['N+2月内示']
    t = forecast.index
    x.index = t
    
    input_data = pd.concat([input_data, x], axis=1)
    input_labels.append(x.name) #説明変数のカラム名の追加
    
    
    #%%
    ##################
    # 月別トレンドの取得
    ##################
    dir_path = os.path.join(base_dir, 'input/month_trend')
    monthly_trend = load_monthly_trend(dir_path) #myfun
    
    ### 各データのトレンド、ピークのラベルを割り振り ###
    months = output_data.index.month #月の取得
    
    n = len(months)
    # 記録領域
    f = pd.DataFrame(np.zeros([n,monthly_trend.shape[1]]), index=output_data.index, columns=monthly_trend.columns)
    
    for k in range(monthly_trend.shape[0]):
        index = np.where(monthly_trend.index[k] == months)[0]
        
        f.iloc[index,:] = monthly_trend.iloc[k,:].values
    
    #--- one-hot表現に変換
    for i in range(f.shape[1]):
        buffer = f.iloc[:,i].dropna()
        columns = np.unique(buffer).astype(int)
        
        one_hot_buf = pd.get_dummies(f.iloc[:,i], columns=columns) #one-hot表現
        
        columns = list(one_hot_buf.columns.astype(int).astype(str))
        for j in range(len(columns)):
            columns[j] = f.columns[i] + '_' + columns[j]
        one_hot_buf.columns = columns
        
        if i >= 1:
            f_one_hot = pd.concat([f_one_hot, one_hot_buf], axis=1)
        else:
            f_one_hot = copy.deepcopy(one_hot_buf)
    
    f = copy.deepcopy(f_one_hot)
    
    
    input_data = pd.concat([input_data, f], axis=1) #説明変数の追加
    #input_labels.append(x.name) #説明変数のカラム名の追加
    input_labels = input_labels + list(f.columns) #説明変数のカラム名の追加
    
    
    #%%
    ##################
    ## 過去の時系列(目的変数)
    ##################
    if 1:
        lags = np.arange(2, 2+3) #ラグ数
        #lags = np.arange(2, 2+4) #ラグ数
        x = gen_lag_data(output_data, lags) #myfun
        
        # カラム名の変更
        columns = list(x.columns)
        for i in range(len(columns)):
            columns[i] = 'y_' + columns[i]
        x.columns = columns
        
        input_data = pd.concat([input_data, x], axis=1)
        input_labels = input_labels + list(x.columns) #説明変数のカラム名の追加
    
    
    ##################
#    lags = np.arange(2, 2+3) #ラグ数
#    #lags = np.arange(2, 2+4) #ラグ数
#    x = gen_lag_data(output_data, lags) #myfun
#    
#    # カラム名の変更
#    columns = list(x.columns)
#    for i in range(len(columns)):
#        columns[i] = 'y_' + columns[i]
#    x.columns = columns
    
    
    ##################
    # 直近の過去データの統計量の算出(目的変数)
    ##################
    windows = [3]
    for i in range(len(windows)):
        w = windows[i]
        buffer = copy.deepcopy(output_data)
        buffer.name = 'y'
        x = calc_stats_in_window(buffer, w) #myfun
        if i >= 1:
            x2 = pd.concat([x2,x], axis=1)
        else:
            x2 = copy.deepcopy(x)
    x2 = x2.shift(2) #タイムスタンプ遷移
    input_data = pd.concat([input_data, x], axis=1)
    input_labels = input_labels + list(x.columns) #説明変数のカラム名の追加
    
    
    ##################
    # 同月1年前/過去のデータ(目的変数)
    ##################
    x = get_1year_before_data(output_data) #myfun
    
    input_data = pd.concat([input_data, x], axis=1)
    input_labels = input_labels + list(x.columns) #説明変数のカラム名の追加
    
    
    
    ##################
    ## 過去の時系列(車両販売計画)
    ##################
    if 0:
        columns = car_sales_data.columns
        col_idx = []
        for i in range(len(columns)):
            if 'N-1' in columns[i]:
                col_idx.append(i)
        
        x1 = car_data[columns[col_idx]]
        
        lags = np.arange(3, 3+5) #ラグ数
        for i in range(x1.shape[1]):
            car_name = x1.columns[i]
            x2 = gen_lag_data(x1.iloc[:,i], lags) #myfun
            
            # カラム名の変更
            columns = list(x2.columns)
            for j in range(len(columns)):
                columns[j] = car_name + '_' + columns[j]
            x2.columns = columns
            
            if i >= 1:
                f = pd.concat([f, x2], axis=1)
            else:
                f = copy.deepcopy(x2)
        
        input_data = pd.concat([input_data, f], axis=1)
        input_labels = input_labels + list(x.columns) #説明変数のカラム名の追加
    
    
    #%% 
    # 目的変数のデータ範囲の指定
    if 0:
        index = np.where(t_start <= output_data.index)[0]
        output_data = output_data.iloc[index]
    
    # 目的変数の形状に合うように、内示をリサイズ
    data = pd.concat([output_data, forecast], axis=1, join_axes=[output_data.index])
    forecast = data.iloc[:,1:data.shape[1]]
    
    
    #%% 説明変数の選択
    ### 目的変数のタイムスタンプに一致するように、説明変数を整形 ###
    x = pd.concat([output_data, input_data], axis=1, join_axes=[output_data.index])
    input_data = x.iloc[:,1:x.shape[1]]
    
    threshold = 3
    x = input_data.fillna(0)
    index = np.where(np.sum(x != 0, axis=0) >= threshold)[0]
    
    input_data = input_data.iloc[:,index]
    
    
    #%% 説明変数の選択(決定係数)
    if 0:
        n_cv = 5 #分割数
        
        x = input_data.fillna(0)
        y = copy.deepcopy(output_data)
        
        # 決定係数による変数選択
        scores, y_pred = feature_select_based_r2(x, y, n_cv) #myfun
        
        threshold = 0.1
        r2_col_idx = np.where(scores >= threshold)[0]
        
        # 例外処理
        if len(r2_col_idx) == 0:
            r2_col_idx = np.arange(x.shape[1])
        
        #print(list(scores.index[r2_col_idx]))
    
    
    #%%
    if 0:
        plt.figure()
        plt.plot(y.values)
        plt.plot(y_pred)
    
    
    #%% 学習/テストデータのキー
    n = output_data.shape[0] #データ長
    valid_key = np.where(output_data.values != 0)[0] #有効なキーの取得
    train_key, test_key = train_test_split(valid_key, test_size=0.4, shuffle=False)
    
    # 例外処理
    if len(valid_key) <= 4:
        print('ERROR : 有効なデータ数が少ない!')
        continue
    
    # 例外処理 : 出荷量一定以上となる月数が少ない場合
    index = np.where(output_data.iloc[test_key] >= 10)[0] #出荷量一定以上となる月のインデックス
    if len(index) < 5:
        print('ERROR : 出荷量一定以上となる月数が少ない!')
        continue
    
    # 例外処理 : 出荷量一定以上となる月数が少ない場合
    index = np.where(output_data.iloc[train_key] >= 10)[0] #出荷量一定以上となる月のインデックス
    if len(index) < 5:
        print('ERROR : 出荷量一定以上となる月数が少ない!')
        continue
    
    
    #%%目的変数のトレンド成分除去
    if 0:
        w = 5
        trend = output_data.rolling(w).mean()
        trend = trend.fillna(0)
        
        y = output_data - trend
        
        plt.figure()
        plt.plot(output_data)
        
        plt.figure()
        plt.plot(y)
    
    
    #%% 予測
    horizon = 2
    n_cv = 5
    
    # 記録領域
    n = len(output_data)
    dim = input_data.shape[1]
    #input_scores = pd.DataFrame(np.zeros([dim,n]), columns=output_data.index, index=input_data.columns)
    input_scores = pd.DataFrame(np.zeros([n,dim]), index=output_data.index, columns=input_data.columns)
    
    if 0:
        ### 対数変換 ###
        log_y = np.log(output_data)
        
        index = np.where(np.isinf(log_y) == 1)[0]
        log_y[index] = 0
        
        ### 差分系列変換 ###
        d_log_y = copy.deepcopy(log_y)
        t = d_log_y.index
        t = [t[k] + relativedelta(months=horizon) for k in range(len(t))]
        d_log_y.index = t
        
        d_log_y = pd.concat([log_y, d_log_y], axis=1, join_axes=[log_y.index]) #結合
        d_log_y = d_log_y.iloc[:,0] - d_log_y.iloc[:,1] #差分
        
        y = copy.deepcopy(d_log_y)
    
    y = copy.deepcopy(output_data)
    y = y.fillna(0)
    t = y.index
    x = input_data.fillna(0)
    
    n = len(test_key) #データ長
    #length = len(y)
    #Y_preds = pd.DataFrame(np.zeros([length,len(mdl_lists)]), columns=mdl_lists, index=y.index) #予測の記録領域
    y_pred = np.zeros(n) #予測の記録領域
    
    train_key1 = list(train_key)
    
    col_idx1 = np.arange(x.shape[1]) #全変数
    #col_idx1 = copy.deepcopy(r2_col_idx) #事前選択された変数
    
    # 例外処理
    if len(col_idx1) == 0:
        col_idx1 = np.arange(x.shape[1])
    
    # テストデータの指定
    for k in range(n):
        print('NO.', str(k), '/', str(n-1))
        
        test_key_ = test_key[k]
        
        # 学習データ追加
        month = (t[test_key_] - t[train_key1[-1]]).days / 30
        if (k >= horizon) and (month >= horizon):
            train_key1.append(test_key[k-horizon])
        
        
        ############
        # 学習データ範囲選択
        ############
        if 0:
            train_idx1, test_idx = train_test_split(train_key1, test_size=0.1, shuffle=False)
            
            length = len(train_idx1) - 4
            scores = np.zeros(length) #記録領域
            scores = pd.DataFrame(np.zeros(length), index=np.arange(length)) #記録領域
            for start in range(length):
                train_idx2 = train_idx1[start:len(train_idx1)]
                
                ym, p = train_test_ridge(x.iloc[:,col_idx1].values, y.values, train_idx2, test_idx, 3) #myfun
                
                # 評価関数の計算
                scores.iloc[start] = r2_score(y[test_idx], ym) #決定係数
                #scores.iloc[start] = rmse(ym, y[test_idx]) # RMSE
            
            #start = np.argmin(scores.values)
            start = np.argmax(scores.values)
            
            train_key2 = train_key1[start:len(train_key1)]
        ############
        # 学習データ範囲選択
        ############
        if 0:
            dist = y.iloc[train_key1] - y.iloc[train_key1[-1]]
            index = np.argsort(abs(dist.values)).astype(int)
            
            if 1:
                index = index[0:24]
                index = np.sort(index)
                train_key2 = [train_key1[index[i]] for i in range(len(index))]
            else:
                train_key2 = copy.deepcopy(train_key1)
        ############
        # 学習データ範囲選択
        ############
        # 直近のデータに対して一定の範囲内に入るデータを学習データとして選択する
        alpha = 3
        y_train = y.iloc[train_key1]
        index, error_flag = select_train_data(y_train, alpha) #myfun
        train_key2 = np.array(train_key1)[index]
        
        # 例外処理
        if len(train_key2) <= 3:
            train_key2 = copy.deepcopy(train_key1)
        
        
        ############
        # 変数選択(elastic netの寄与率)
        ############
        if 0:
            _, p = train_test_en(x.iloc[:,col_idx1].values, y.values, train_key2, 0, n_cv) #myfun
            
            index, p_rate = var_select_based_contr_rate(p, 1) #myfun
            
            if len(index) != 0:
                col_idx2 = col_idx1[index]
            else:
                col_idx2 = copy.deepcopy(col_idx1)
            
            input_scores.iloc[test_key_,:] = p_rate #store
        #col_idx2 = copy.deepcopy(col_idx1)
        ############
        # 変数選択(SHAP)
        ############
        if 1:
            train_idx = copy.deepcopy(train_key2)
            #---
            #test_idx = copy.deepcopy(train_key2)
            #test_idx = np.append(train_key2, test_key_)
            test_idx = np.arange(test_key_-5, test_key_)
            #test_idx = copy.deepcopy([test_key_])
            
            # モデル構築
            _, p, clf = train_test_randomForest(x.iloc[:,col_idx1].values, y.values, train_idx, 0, n_cv) #myfun
            
            # 正規化
            x_scaler = StandardScaler()
            x_scaler.fit(x.iloc[train_key2,:].values)
            x_ = x_scaler.transform(x.values)
            x_ = pd.DataFrame(x_, index=x.index, columns=x.columns) #データフレームに変換
            
            # SHAP値の算出
            explainer = shap.TreeExplainer(clf)
            
            #shaps = np.zeros([len(test_idx), x_.shape[1]]) #記録領域
            
            #shap_values = explainer.shap_values(x.iloc[[test_key_],col_idx1])
            #shaps[:,col_idx1] = explainer.shap_values(x_.iloc[test_idx,col_idx1])
            shaps = explainer.shap_values(x_.iloc[test_idx,col_idx1])
            shaps = pd.DataFrame(shaps, columns=x_.columns[col_idx1]) #データフレームに変換
            
            # スコア算出
            scores = abs(shaps.mean())
            #scores = pd.DataFrame(scores, columns=['score']) #データフレームに変換
            #scores.insert(0, 'input', scores.index)
            #scores.index = np.arange(scores.shape[0]) #indexのリセット
            
            input_scores.iloc[test_key_,col_idx1] = scores.values #store
            
            states = 'threshold'
            #states = 'non_zero'
            #states = 'ranking'
            if states == 'threshold':
                ### shap閾値以上の変数選択 ###
                threshold = 0.005
                #threshold = 0.01
                index = np.where(scores >= threshold)[0]
            elif states == 'non_zero':
                index = np.where(scores != 0)[0]
            elif states == 'ranking':
                ### shap上位の変数選択 ###
                index = scores.argsort().values[::-1]
                index = index[0:10]
            else:
                index = np.arange(x.shape[1])
            
            if len(index) != 0:
                col_idx2 = col_idx1[index]
            else:
                col_idx2 = copy.deepcopy(col_idx1)
            #col_idx2 = copy.deepcopy(col_idx1)
        ############
        # 変数選択(ランダムフォレスト)
        ############
        if 0:
            #if (k%2 == 0) or (k == 0):
            if 1:
                clf = RFR(
                    n_estimators=50, 
                    criterion='mse', 
                    max_depth=7, 
                    max_features='sqrt', 
                    n_jobs=3,
                    verbose=False
                    )
                
                feat_selector = BorutaPy(clf, 
                                 n_estimators='auto',  # 特徴量の数に比例して、木の本数を増やす
                                 two_step=False,
                                 verbose=1, # 0: no output,1: displays iteration number,2: which features have been selected already
                                 alpha=0.01, # 有意水準
                                 max_iter=40, # 試行回数
                                 random_state=0
                                )
                
                x_train = x.iloc[train_key2,col_idx1].values
                y_train = y.iloc[train_key2].values
                
                ### 正規化 ###
                x_scaler = StandardScaler()
                y_scaler = StandardScaler()
                
                x_scaler.fit(x_train)
                y_scaler.fit(y_train.reshape(-1,1))
                
                x_train = x_scaler.transform(x_train)
                y_train = y_scaler.transform(y_train.reshape(-1,1))
                
                # 変数選択
                feat_selector.fit(x_train, y_train)
                
                index = np.where(feat_selector.support_ == 1)[0]
                col_idx2 = col_idx1[index]
                
                print('選択された説明変数の数: ', len(col_idx2))
                
                if len(index) != 0:
                    col_idx2 = col_idx1[index]
                else:
                    col_idx2 = copy.deepcopy(col_idx1)
                input_scores.iloc[test_key_,col_idx2] = 1 #store
        ############
        # 変数選択(決定係数)
        ############
        if 0:
            scores, _ = feature_select_based_r2(x.iloc[train_key2,col_idx1], y.iloc[train_key2], n_cv) #myfun
            
            threshold = 0.1
            index = np.where(scores >= threshold)[0]
            
            if len(index) != 0:
                col_idx2 = col_idx1[index]
            else:
                col_idx2 = copy.deepcopy(col_idx1)
        
        
        ############
        # 正規化
        ############
        x_norm, _ = normalize(x.values, train_key2) #myfun
        y_norm, y_scaler = normalize(y.values, train_key2) #myfun
        
        x_norm = pd.DataFrame(x_norm, index=x.index, columns=x.columns)
        y_norm = pd.Series(y_norm, index=y.index, name=y.name)
        
        
        ############
        # 最適モデル選択
        ############
        ### パラメータリストの生成 ###
        params = {} #記録領域
        params['ridge'] = 10 ** np.arange(-2, 1, 0.1)
        params['en'] = {
                'l1_ratio': [.05, .15, .5, .7, .9, .95, .99, 1],
                'n_alphas': 20,
                }
        params['rf'] = {
                'max_depth': [3,5],
                'n_estimators': [80],
                'min_samples_split': [2],
                'min_samples_leaf': [3],
                'bootstrap': [True],
                }
        params['rf'] = {
                'max_depth': 5,
                'n_estimators': 80,
                'min_samples_split': 2,
                'min_samples_leaf': 3,
                'bootstrap': True,
                }
        params['xgb'] = {
                }
        
        mdl_lists = list(params.keys())
        
        
        # テストデータのサブセット生成
        length = round(len(train_key2) / 3)
        TestKeys = list(split_list(list(train_key2), length)) #myfun
        
        dim = len(mdl_lists)
        length = len(y)
        yhat = pd.DataFrame(np.zeros([length,dim]), columns=mdl_lists, index=y.index) #記録領域
        
        # 学習データのサブセットの指定
        for test_id in range(len(TestKeys)):
            test_idx = TestKeys[test_id]
            train_idx = list(set(train_key2) - set(test_idx))
            
            x_train = x_norm.iloc[train_idx,col_idx2].values
            y_train = y_norm.iloc[train_idx].values
            
            # 例外処理
            n_cv_ = copy.deepcopy(n_cv)
            if n_cv >= len(train_idx):
                n_cv_ = len(train_idx)
            
            mdls = {} #記録領域
            col_name = 'ridge'
            mdls[col_name] = RidgeCV(alphas=params[col_name], cv=n_cv_)
            col_name = 'en'
            mdls[col_name] = ElasticNetCV(**params[col_name], n_jobs=2, cv=n_cv_)
            col_name = 'rf'
            mdls[col_name] = RFR(**params[col_name], n_jobs=2, random_state=2525)
            col_name = 'xgb'
            mdls[col_name] = XGBRegressor(**params[col_name], n_jobs=2, random_state=2525)
            
            for i in range(len(mdl_lists)):
                clf = mdls[mdl_lists[i]]
                clf.fit(x_train, y_train)
                mdls[mdl_lists[i]] = clf #store
            
            x_test = x_norm.iloc[test_idx,col_idx2].values
            
            for i in range(len(mdl_lists)):
                col_name = mdl_lists[i]
                clf = mdls[col_name]
                tmp = clf.predict(x_test)
                yhat[col_name].iloc[test_idx] = y_scaler.inverse_transform(tmp) #非正規化
        
        # 各モデルの予測精度の算出
        scores = np.zeros(len(mdl_lists)) #記録領域
        for i in range(len(mdl_lists)):
            scores[i] = rmse(yhat.iloc[train_key2,i].values, y.iloc[train_key2].values) #myfun
        best_idx = scores.argmin()
        
        
        ############
        # 予測
        ############
        x_train = x_norm.iloc[train_key2,col_idx2].values
        y_train = y_norm.iloc[train_key2].values
        
        x_test = x_norm.iloc[test_key_,col_idx2].values.reshape(1,-1)
        
        # 例外処理
        n_cv_ = copy.deepcopy(n_cv)
        if n_cv >= len(train_key2):
            n_cv_ = len(train_key2)
        
        mdls = {} #記録領域
        col_name = 'ridge'
        mdls[col_name] = RidgeCV(alphas=params[col_name], cv=n_cv_)
        col_name = 'en'
        mdls[col_name] = ElasticNetCV(**params[col_name], n_jobs=2, cv=n_cv_)
        col_name = 'rf'
        mdls[col_name] = RFR(**params[col_name], n_jobs=2, random_state=2525)
        col_name = 'xgb'
        mdls[col_name] = XGBRegressor(**params[col_name], n_jobs=2, random_state=2525)
        
        
        ### train ###
        mdl_name = mdl_lists[best_idx]
        clf = mdls[mdl_name]
        clf.fit(x_train, y_train)
        mdls[mdl_name] = clf #store
        
        
        ### test ###
        clf = mdls[mdl_name]
        tmp = clf.predict(x_test)
        y_pred[k] = y_scaler.inverse_transform(tmp) #非正規化
    
    
    #%% データ範囲を超える場合、内示で置換
    ### データ範囲の算出 ###
    w = 12
    _, index1 = replace_forecast_in_out_of_range(y, y_pred, forecast['N+2月内示'], valid_key, test_key, horizon, w) #myfun
    
    
    ### 予測値が負の場合、0で置換 ###
    index2 = np.where(y_pred < 0)[0]
    
    index = np.append(index1,index2)
    index = np.sort(np.unique(index))
    
    index = copy.deepcopy(index2)
    
    y_pred2 = copy.deepcopy(y_pred)
    y_pred2[index] = forecast['N+2月内示'].iloc[test_key[index]]
    
    
    #%% 評価
    y = copy.deepcopy(output_data)
    
    columns = ['内示誤差率 [%]', '予測誤差率 [%]']
    err_rate = pd.DataFrame(np.zeros([1,len(columns)]), columns=columns, index=[output_name]) #記録領域
    
    tmp = calc_mean_error_rate(y.iloc[test_key].values, forecast['N+2月内示'].iloc[test_key].values) #my fun
    err_rate['内示誤差率 [%]'] = tmp.mean()
    
    tmp = calc_mean_error_rate(y.iloc[test_key], y_pred2) #my fun
    err_rate['予測誤差率 [%]'] = tmp.mean()
    
    
    #### 平均誤差率の履歴保存 ###
    for i in range(2):
        col_name = err_rate.columns[i]
        all_err_rate[col_name].iloc[output_id] = err_rate[col_name].iloc[0]
    
    ### 表示 ###
    for i in range(err_rate.shape[1]):
        col_name = err_rate.columns[i]
        print(col_name, '=', err_rate.iloc[0,i], '%')
    
    
    #%% save
    ### 全品目の平均誤差率の保存 ###
    fname = os.path.join(output_dir, 'err_rate.csv')
    all_err_rate.to_csv(fname, encoding='cp932')
    
    ### 平均誤差率の保存 ###
    output_dir_ = os.path.join(output_dir, 'csv')
    if os.path.isdir(output_dir_) == False:
        os.makedirs(output_dir_)
    
    fname = 'NO' + str(output_id) + '_'+  output_name + '.csv'
    fname = os.path.join(output_dir_, fname)
    err_rate.to_csv(fname, encoding='cp932')
    
    
    ### 予測結果の明細の保存 ###
    output_dir_ = os.path.join(output_dir, 'detail_csv')
    if os.path.isdir(output_dir_) == False:
        os.makedirs(output_dir_)
    
    fname = 'NO' + str(output_id) + '_'+  output_name + '.csv'
    fname = os.path.join(output_dir_, fname)
    
    ### 結果明細の出力 ###
    res = output_result_detail(fname, output_data, forecast['N+2月内示'], y_pred2, test_key) #myfun
    
    ### shapスコアの保存 ###
    output_dir_ = os.path.join(output_dir, 'input_score')
    if os.path.isdir(output_dir_) == False:
        os.makedirs(output_dir_)
    
    fname = 'NO' + str(output_id) + '_'+  output_name + '.csv'
    fname = os.path.join(output_dir_, fname)
    input_scores.to_csv(fname, encoding='cp932')
    
    
    #%%
    ###################
    # グラフ表示 : 実績/予測の時系列
    ###################
    save_flag = 1
    fontsize = 15
    figsize = (12,10)
    
    ### 実績/内示/予測(テスト期間) ###
    # set output filename
    fname = 'NO' + str(output_id) + '_' + output_name + '.png'
    fname = os.path.join(output_dir, 'fig', fname)
    
    t = y.index
    
    title_name = 'NO.' + str(output_id) + '  [' + col_name + ' : ' + f'{err_rate.iloc[0,1]:.1f}' + '%]   ' + '[内示 : ' + f'{err_rate.iloc[0,0]:.1f}' + ' %]'
    
    plot_time_series(fname, t[test_key], y.iloc[test_key], y_pred2, forecast['N+2月内示'].iloc[test_key], output_name, title_name, None, 1, fontsize, figsize, save_flag)
    plt.close()
    
    
    #%%
    ### 実績/内示(検証期間) ###
    figsize = (16,10)
    
    # make output directory
    output_dir_ = os.path.join(output_dir, 'raw_test')
    if os.path.isdir(output_dir_) == False:
        os.makedirs(output_dir_)
    
    # set output filename
    fname = 'NO' + str(output_id) + '_' + output_name + '.png'
    fname = os.path.join(output_dir_, fname)
    
    n = len(t) #データ長
    tmp = np.zeros(n)
    tmp[tmp == 0] = np.nan
    plot_time_series(fname, t, y.values, tmp, forecast['N+2月内示'].values, output_name, '', None, 1, fontsize, figsize, save_flag)
    plt.close()
    
    #%% 目的/説明変数のプロット
    save_flag = 1
    fontsize = 16
    figsize = (12,16)
    
    
    # スコアの平均値
    index = np.where(input_scores.iloc[:,0] != 0)[0]
    #scores = input_scores.iloc[index,:].mean(axis=0)
    #---
    scores = input_scores.iloc[-1,:]
    
    threshold = 0.01
    input_list = np.where(scores >= threshold)[0]
    
    # 変数選択(ランダムフォレスト)
    if 0:
        input_list = np.where(input_scores.iloc[-1,:] == 1)[0]
    
    
    #length = input_data.shape[1]
    length = len(input_list)
    
    ranking = np.argsort(scores.values)[::-1]
    #tmp = scores.iloc[index]
    
    # 説明変数の指定
    for input_id_ in range(length):
        print('input NO.', str(input_id_), '/', str(length-1))
        input_id = input_list[input_id_]
        input_name = input_data.columns[input_id]
        data = pd.concat([output_data, input_data.iloc[:,input_id]], axis=1)
        t = data.index
        
        ranking_id = np.where(input_id == ranking)[0][0]
        title_name = 'SHAP score = ' + str(scores.iloc[input_id]) + '  ranking NO.' + str(ranking_id)
        
        # make output directory
        dir_name = 'NO' + str(output_id) + '_' + output_name
        output_dir_ = os.path.join(output_dir, 'inputs', dir_name)
        if os.path.isdir(output_dir_) == False:
            os.makedirs(output_dir_)
        
        # set output filename
        fname = 'NO' + str(output_id) + '_' + output_name + '__NO' + str(input_id) + '_' + input_name + '.png'
        fname = os.path.join(output_dir_, fname)
        
        make_subplot(fname, t, data, title_name, fontsize, figsize, save_flag) #myfun
        plt.close()




