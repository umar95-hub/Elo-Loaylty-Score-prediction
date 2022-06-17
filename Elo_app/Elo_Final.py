# -*- coding: utf-8 -*-
"""
Created on Thu May 12 12:47:34 2022

@author: faruk
"""

import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb

import datetime
import pickle
from tqdm import tqdm
warnings.filterwarnings("ignore")

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer



from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb



from sklearn.metrics import mean_squared_error
from math import sqrt

import gc
import warnings
warnings.filterwarnings('ignore')


#from sklearn.metrics import mean_squared_error
#from math import sqrt

import os

#https://www.kaggle.com/fabiendaniel/elo-world
#Function to load data into pandas and reduce memory usage

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

#https://www.geeksforgeeks.org/python-pandas-series-dt-date/
def getFeaturesFromTrainAndTest(data):

    min_dte = data['first_active_month'].dt.date.min()

    #Time elapsed since first purchase
    data['time_elapsed'] = (data['first_active_month'].dt.date - min_dte).dt.days

    #Breaking first_active_month in year and month
    data['month'] = data['first_active_month'].dt.month
    data['year'] = data['first_active_month'].dt.year
    data['day'] = data['first_active_month'].dt.day
    
    
def get_date_feature(df_m):

  df = df_m.copy()
  del df_m
  df['purchase_date'] = pd.to_datetime(df['purchase_date'])

  df['m_op'] = df['purchase_date'].dt.month
  df['y_op'] = df['purchase_date'].dt.year
  
  df['week_day'] = df['purchase_date'].dt.day.apply(lambda x : 0 if x > 4 else 1)
  df['weekend'] = df['purchase_date'].dt.day.apply(lambda x : 0 if x < 4 else 1)

  df['days'] = ((df['purchase_date'] - datetime.datetime(2011,11,1))).dt.days

  df['month_diff'] = ((df['purchase_date'] - datetime.datetime(2011,11,1))).dt.days//30 #Month diff from the date of first card activation
  
  df['month_diff'] += df['month_lag']
    
  df = pd.get_dummies(df, columns =['m_op','y_op'])
 
  agg_func_1 = {
      'week_day' : ['sum','mean'],
      'weekend' : ['sum','mean'],
      'month_diff' : ['mean','min','max','std'],
      'month_lag' : ['mean','min','max','std']     
  }

  agg_func_2 = { i : ['sum', 'mean'] for i in df.columns if ('m_op' in i) or ('y_op' in i)}
  
  df_1 = df.groupby('card_id').agg(agg_func_1)
  df_1.columns = [ "_".join(col).strip() for col in df_1.columns]
  df_1 = df_1.reset_index()

  df_2 = df.groupby('card_id').agg(agg_func_2)
  df_2.columns = ["_".join(col).strip() for col in  df_2.columns]
  df_2 = df_2.reset_index()

  df = df_1.merge(df_2, how = 'left', on = 'card_id')
  return df

def get_purchase_features(df_main):
  df = df_main.copy()
  del df_main
  agg_d = {'purchase_amount' : ['sum']}

  df['purchase_date'] = pd.to_datetime(df['purchase_date'])

  df['m_op'] = df['purchase_date'].dt.month
  df['y_op'] = df['purchase_date'].dt.year

  df_m = df.groupby(['card_id', 'm_op']).agg(agg_d)

  #fm = pd.get_dummies(nm, columns = ['m_op'])
  df_m.columns = ["_".join(col).strip() for col in df_m.columns]
  df_m = df_m.reset_index()
  #print('Pass')
  df_m = pd.get_dummies(df_m, columns = ['m_op'])

  ar = np.array(df_m.drop(columns = ['card_id','purchase_amount_sum']))

  ar1 = np.array(df_m['purchase_amount_sum'])
  ar1 = ar1.reshape(-1,1)

  l1 = ['m_op_'+ str(i) for i in range(1,13)]

  df_m[l1] = ar1*ar
  df_m = df_m.drop(columns = ['purchase_amount_sum'])
  df_m = df_m.groupby('card_id').agg(['sum','mean'])

  #print('Pass')
  ######################################################

  df_y = df.groupby(['card_id', 'y_op']).agg(agg_d)

  #fm = pd.get_dummies(nm, columns = ['m_op'])
  df_y.columns = ["_".join(col).strip() for col in df_y.columns]
  df_y = df_y.reset_index()

  df_y = pd.get_dummies(df_y, columns = ['y_op'])

  ar = np.array(df_y.drop(columns = ['card_id','purchase_amount_sum']))

  ar1 = np.array(df_y['purchase_amount_sum'])
  ar1 = ar1.reshape(-1,1)

  l1 = ['y_op_2017','y_op_2018']

  df_y[l1] = ar1*ar
  df_y = df_y.drop(columns = ['purchase_amount_sum'])
  df_y = df_y.groupby('card_id').agg(['sum','mean'])

  df = df_y.merge(df_m, on = 'card_id', how = 'left')

  df.columns = ["_".join(col).strip() for col in df.columns]
  df.reset_index(inplace = True)
  return df

def agg_merchant(df):

  
  dic_agg = {i:['mean'] for i in df.columns if (i not in ['merchant_id','category_4'])}
  dic_agg['most_recent_sales_range'] = ['nunique']
  dic_agg['most_recent_purchases_range'] = ['nunique']
  dic_agg['merchant_group_id'] = ['nunique']
  dic_agg['category_4'] = ['nunique','mean']


  agg = df.groupby(['merchant_id']).agg(dic_agg)
  
  agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
  agg.reset_index(inplace=True)

  return agg

def aggregate_transactions(df):

    agg_func = {
    'category_1': ['sum', 'mean'],

    'category_3_0.0': ['mean','sum'],
    'category_3_1.0': ['mean','sum'],
    'category_3_2.0': ['mean','sum'],

    'category_2_1.0': ['mean','sum'],
    'category_2_2.0': ['mean','sum'],
    'category_2_3.0': ['mean','sum'],
    'category_2_4.0': ['mean','sum'],
    'category_2_5.0': ['mean','sum'],

    'authorized_flag_0' :['mean','sum'],
    'authorized_flag_1' :['mean','sum'],

    'merchant_id': ['nunique'],
    'merchant_category_id': ['nunique',mode],

    'state_id': ['nunique',mode],
    'city_id': ['nunique',mode],
    'subsector_id': ['nunique',mode],

    'purchase_amount': ['sum', 'mean', 'max', 'min', std],
    'installments': ['sum', 'mean', 'max', 'min', std],  
    
    'numerical_1_mean' : ['sum', 'mean', 'max', 'min',std],
    'numerical_2_mean' : ['sum', 'mean', 'max', 'min',std],
    

    'merchant_group_id_nunique' : ['mean','sum'],
    'most_recent_sales_range_nunique' : ['mean','sum'],
    'most_recent_purchases_range_nunique' : ['mean','sum'],


    'avg_sales_lag3_mean' : ['sum', 'mean',std],
    'avg_purchases_lag3_mean':  ['sum', 'mean',std],
    'active_months_lag3_mean':['sum', 'mean',std],

    'avg_sales_lag6_mean':['sum', 'mean',std],
    'avg_purchases_lag6_mean':['sum', 'mean',std],
    'active_months_lag6_mean': ['sum', 'mean',std],

    'avg_sales_lag12_mean':['sum', 'mean', std],
    'avg_purchases_lag12_mean': ['sum', 'mean', std],
    'active_months_lag12_mean': ['sum', 'mean',std],

    'category_4_nunique' : ['sum','mean'],
    'category_4_mean' : ['sum', 'mean']

     }
    
    df_n = df.groupby('card_id').agg(agg_func)
    
    df_tr = (df.groupby('card_id')\
            .size()\
            .reset_index(name='transactions_count'))

    df_n.columns = ['_'.join(col).strip() for col in df_n.columns.values]

    df_n = df_n.merge(df_tr, how = 'left', on = 'card_id')

    

    return df_n   



def aggregate_per_month(df):
    grouped = df.groupby(['card_id', 'month_lag'])

    agg_func = {
            'purchase_amount': ['count', 'sum', 'mean', 'min', 'max',std],
            'installments': ['count', 'sum', 'mean', 'min', 'max',std],
            }

    intermediate_group = grouped.agg(agg_func)
    intermediate_group.columns = ['_'.join(col).strip() for col in intermediate_group.columns.values]
    intermediate_group.reset_index(inplace=True)

    final_group = intermediate_group.groupby('card_id').agg(['mean', 'std'])
    final_group.columns = ['_'.join(col).strip() for col in final_group.columns.values]
    final_group.reset_index(inplace=True)
    
    return final_group

import statistics as st
from scipy import stats as s

def std(x):
  return np.std(x)

def mode(x):
  return int(s.mode(x)[0])
def mode_c(x):
  return x.value_counts().index[0]

def agg_merchant(df):

  
  dic_agg = {i:['mean'] for i in df.columns if (i not in ['merchant_id','category_4'])}
  dic_agg['most_recent_sales_range'] = ['nunique']
  dic_agg['most_recent_purchases_range'] = ['nunique']
  dic_agg['merchant_group_id'] = ['nunique']
  dic_agg['category_4'] = ['nunique','mean']


  agg = df.groupby(['merchant_id']).agg(dic_agg)
  
  agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
  agg.reset_index(inplace=True)

  return agg

def final_table():

  #Importing Datasets

  train_table  = reduce_mem_usage(pd.read_csv('train.csv',parse_dates = ["first_active_month"]))

  '''test_table  = reduce_mem_usage(pd.read_csv('test.csv',parse_dates = ["first_active_month"]))
  test_table = test_table.fillna(test_table.first_active_month[17])'''

  historical_transaction = reduce_mem_usage(pd.read_csv('historical_transactions.csv'))
  new_merchants_trasaction = reduce_mem_usage(pd.read_csv("new_merchant_transactions.csv"))

  merchants = reduce_mem_usage(pd.read_csv("/content/merchants.csv"))


  #Imputation
  new_merchants_trasaction.category_1 = new_merchants_trasaction.category_1.map({'Y':1, 'N':0})
  new_merchants_trasaction.category_3 = new_merchants_trasaction.category_3.map({'A':0, 'B':1, 'C':2})
  new_merchants_trasaction.authorized_flag = new_merchants_trasaction.authorized_flag.map({'Y':1, 'N':0})

  historical_transaction.category_1 = historical_transaction.category_1.map({'Y':1, 'N':0})
  historical_transaction.category_3 = historical_transaction.category_3.map({'A':0, 'B':1, 'C':2})
  historical_transaction.authorized_flag = historical_transaction.authorized_flag.map({'Y':1, 'N':0})

  missing_id = historical_transaction.merchant_category_id.loc[historical_transaction.merchant_id.isnull()].unique()

  missing_id_nm = new_merchants_trasaction.merchant_category_id.loc[new_merchants_trasaction.merchant_id.isnull()].unique()

  
  for i in tqdm(missing_id):
      
      mask = historical_transaction.merchant_category_id == i

      value = historical_transaction.loc[mask,"merchant_id"].value_counts().index[0]
      
      historical_transaction.loc[mask,"merchant_id"] = historical_transaction.loc[mask,"merchant_id"].fillna(value)

  #imputation Merchant ids in new_merchant

  for i in tqdm(missing_id_nm):
      
      mask = new_merchants_trasaction.merchant_category_id == i
      value = new_merchants_trasaction.loc[mask,"merchant_id"].value_counts().index[0]
      
      #new_merchants_trasaction.merchant_id.loc[new_merchants_trasaction.merchant_category_id == i].fillna(value, inplace = True)
      
      new_merchants_trasaction.loc[mask,"merchant_id"] = new_merchants_trasaction.loc[mask,"merchant_id"].fillna(value)
        

  clean_nm = new_merchants_trasaction.drop(columns = ['card_id','purchase_date','merchant_id'])
  #clm = list(new_merchants_trasaction.drop(columns = ['card_id','purchase_date','merchant_id']).columns)
  #print(" Index numbers for category2 , 3 in new merchant",clm.index('category_2'),clm.index('category_3'))

  clean_ht = historical_transaction.drop(columns = ['card_id','purchase_date','merchant_id'])
  #clm = list(historical_transaction.drop(columns = ['card_id','purchase_date','merchant_id']).columns)
  #print(" Index numbers for category2 , 3 in historical data",clm.index('category_2'),clm.index('category_3'))
  
  imp = IterativeImputer(max_iter= 15, random_state=0)

  clean_nm = np.round(imp.fit_transform(clean_nm))

  clean_ht = np.round(imp.fit_transform(clean_ht))

  new_merchants_trasaction['category_3'] = clean_nm[:,4]
  new_merchants_trasaction['category_2'] = clean_nm[:,8]

  historical_transaction['category_3'] = clean_ht[:,4]
  historical_transaction['category_2'] = clean_ht[:,8]

  #Rounding off
  new_merchants_trasaction['category_3'] = new_merchants_trasaction['category_3'].apply(lambda x : 2 if x > 2 else x)
  new_merchants_trasaction['category_2'] = new_merchants_trasaction['category_2'].apply(lambda x : 5 if x > 5 else x)

  del clean_ht
  del clean_nm

  transaction = historical_transaction.append(new_merchants_trasaction, ignore_index=True)

  del historical_transaction
  del new_merchants_trasaction

  transaction = pd.get_dummies(transaction, columns=['category_2', 'category_3','authorized_flag'])

  #Merchant DATASET Imputation

  merchants = merchants[['merchant_id','numerical_1', 'numerical_2','merchant_group_id',
       'most_recent_sales_range', 'most_recent_purchases_range',
       'avg_sales_lag3', 'avg_purchases_lag3', 'active_months_lag3',
       'avg_sales_lag6', 'avg_purchases_lag6', 'active_months_lag6',
       'avg_sales_lag12', 'avg_purchases_lag12', 'active_months_lag12',
       'category_4']]

  na_imputer = lambda x: x.fillna(x.mean())
  inf_imputer = lambda x: x.replace([np.inf, -np.inf], x.value_counts().index[0])

  merchants[['avg_purchases_lag3']] = merchants[['avg_purchases_lag3']].apply(inf_imputer)
  merchants[['avg_purchases_lag6']] = merchants[['avg_purchases_lag6']].apply(inf_imputer)
  merchants[['avg_purchases_lag12']] = merchants[['avg_purchases_lag12']].apply(inf_imputer)

  merchants[['avg_sales_lag3']] = merchants[['avg_sales_lag3']].apply(na_imputer)
  merchants[['avg_sales_lag6']] = merchants[['avg_sales_lag6']].apply(na_imputer)
  merchants[['avg_sales_lag12']] = merchants[['avg_sales_lag12']].apply(na_imputer)

  merchants.category_4 = merchants.category_4.apply( lambda x : 1 if x == 'Y' else 0)

  # Get Features

  tr_purchase_f = get_purchase_features(transaction)
  tr_date_f = get_date_feature(transaction)

  merchants = agg_merchant(merchants)

  transaction = transaction.merge(merchants, on = 'merchant_id', how = 'left')
  del merchants

  nm_tra_agg = aggregate_transactions(transaction)
  nm_on_mlag = aggregate_per_month(transaction)

  del transaction

  #imputing na, inf values

  nm_tra_agg = nm_tra_agg.replace([np.inf, - np.inf], np.nan)
  list_na = [i for i in range(len(nm_tra_agg.columns)) if nm_tra_agg.isnull().any()[i] == True]
  for i in list_na:
    nm_tra_agg.iloc[:,i] = nm_tra_agg.iloc[:,i].fillna(nm_tra_agg.iloc[:,i].mean())

  #Few more features
  nm_tra_agg['purchase_importance_sum'] = nm_tra_agg['installments_sum']*np.log(abs(nm_tra_agg.transactions_count/nm_tra_agg.purchase_amount_sum))

  nm_tra_agg['purchase_importance_mean'] = nm_tra_agg['installments_mean']*np.log(abs(nm_tra_agg.transactions_count/nm_tra_agg.purchase_amount_mean))

  #nm_tra_agg['purchase_importance_std'] = nm_tra_agg['installments_std']*np.log(abs(nm_tra_agg.transactions_count/nm_tra_agg.purchase_amount_std))

  nm_tra_agg['purchase_importance_min'] = nm_tra_agg['installments_min']*np.log(abs(nm_tra_agg.transactions_count/nm_tra_agg.purchase_amount_min))

  nm_tra_agg['purchase_importance_max'] = nm_tra_agg['installments_mean']*np.log(abs(nm_tra_agg.transactions_count/nm_tra_agg.purchase_amount_max))



  # Mean hot encoding of categorical values(f1, f2, f3)

  train_table['rare_value'] = train_table['target'].apply(lambda x: 1 if x<= -30 else 0)

  inf_imputer = lambda x: x.replace([np.inf, -np.inf], x.value_counts().index[0])

  for i in ['feature_1', 'feature_2', 'feature_3']:

      rare_data_mean = train_table.groupby([i])['rare_value'].mean()

      train_table[i] = train_table[i].map(rare_data_mean)

      #test_table[i] = test_table[i].map(rare_data_mean)

  

  train_final = train_table.merge(nm_tra_agg, how = 'left', on = 'card_id')
  #test_final = test_table.merge(nm_tra_agg, how = 'left', on = 'card_id')

  del nm_tra_agg

  train_final = train_final.merge(nm_on_mlag, how = 'left', on = 'card_id')
  #test_final = test_final.merge(nm_on_mlag, how = 'left', on = 'card_id')

  del nm_on_mlag

  train_final = train_final.merge(tr_purchase_f, how = 'left', on = 'card_id')
  #test_final = test_final.merge(tr_purchase_f, how = 'left', on = 'card_id')

  del tr_purchase_f

  train_final = train_final.merge(tr_date_f, how = 'left', on = 'card_id')
  #test_final = test_final.merge(tr_date_f, how = 'left', on = 'card_id')

  del tr_date_f

  return train_final


###Final Function_1

def function_1 (data):

    destination_1 = 'final_table.pkl'
    
    if os.path.isfile(destination_1): 
      
        final_table = reduce_mem_usage(pd.read_pickle('final_table.pkl'))
    
    else:
    
        try:
      
          final_table = final_table()
      
          pickle.dump((final_table), open('final_table.pkl','wb'))
      
          final_table = reduce_mem_usage(pd.read_pickle('final_table.pkl'))
      
        except Exception as e:
            print('Error:',e)
    
    #importing best model
    
    try :
        
        model = pickle.load(open('kf_fold.pkl', 'rb'))
    
    except Exception as e:
        
        print('Error:',e)
    
    id = data.card_id.values.tolist()
    
    #print("pass")
    index = [final_table.loc[final_table.card_id == i].index.item() for i in id]
    
    x = final_table.loc[index,:]
    
    x = x.drop(columns = ['first_active_month','card_id','rare_value','target'])
    
    prediction = model.predict(x)
    
    return prediction


    