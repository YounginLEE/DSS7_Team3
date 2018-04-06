import numpy as np
import pandas as pd
from sklearn.preprocessing import *

def FN_dummies(df, test): # 메모리 문제로 돌리지를 못함
    both = set(test['FinelineNumber'].unique()) & set(df['FinelineNumber'].unique())
    df = df[df['FinelineNumber'].isin(both)] # train과 test 둘다 있는 것만으로 모델링

    df['FinelineNumber']=df['FinelineNumber'].astype(str) # get_dummies 조건
    test['FinelineNumber']=test['FinelineNumber'].astype(str)

    FN = pd.get_dummies(df[['VisitNumber','FinelineNumber']]).groupby('VisitNumber').sum()
    T_FN = pd.get_dummies(df[['VisitNumber','FinelineNumber']]).groupby('VisitNumber').sum()

    not_in_train = list(set(T_FN.columns) - set(FN.columns))
    if not_in_train:
        T_FN.drop(not_in_train, axis=1, inplace=True) # 0으로 설정된 컬럼 삭제 (train에 없음)

    FN.reset_index(inplace=True)
    T_FN.reset_index(inplace=True)

    return FN, T_FN

def DD_dummies(df, test):
    both = set(test['DepartmentDescription'].unique()) & set(df['DepartmentDescription'].unique())
    df = df[df['DepartmentDescription'].isin(both)] # train과 test 둘다 있는 것만으로 모델링
    df.reset_index(drop=True, inplace=True)

    DD = pd.get_dummies(df[['VisitNumber','DepartmentDescription']]).groupby('VisitNumber').sum()
    T_DD = pd.get_dummies(test[['VisitNumber','DepartmentDescription']]).groupby('VisitNumber').sum()

    not_in_train = list(set(T_DD.columns) - set(DD.columns))
    if not_in_train:
        T_DD.drop(not_in_train, axis=1, inplace=True)

    DD.reset_index(inplace=True)
    T_DD.reset_index(inplace=True)

    return DD, T_DD

def editor(df, test, type='DD'):
    df['refund'] = df['ScanCount'].apply(lambda x: -x if x < 0 else 0)
    df['purchase'] = df['ScanCount'].apply(lambda x: x if x > 0 else 0)
    test['refund'] = test['ScanCount'].apply(lambda x: -x if x < 0 else 0)
    test['purchase'] = test['ScanCount'].apply(lambda x: x if x > 0 else 0)

    df1 = df[['VisitNumber','refund','purchase']].groupby('VisitNumber').sum()
    df2 = pd.get_dummies(df[['VisitNumber','TripType','Weekday']]).groupby('VisitNumber').mean()
    test1 = test[['VisitNumber','refund','purchase']].groupby('VisitNumber').sum()
    test2 = pd.get_dummies(test[['VisitNumber', 'Weekday']]).groupby('VisitNumber').mean()

    df1.reset_index(inplace=True)
    df2.reset_index(inplace=True)
    test1.reset_index(inplace=True)
    test2.reset_index(inplace=True)

    df1 = pd.merge(df1, df2, on='VisitNumber')
    test1 = pd.merge(test1, test2, on='VisitNumber')

    #bring dummies
    if type == 'DD':
        df, test = DD_dummies(df, test)

    elif type == 'FN':
        df, test = FN_dummies(df, test)

    elif type == 'all':
        DD, Dtest = DD_dummies(df, test)
        FN, Ftest = FN_dummies(df, test)
        df = pd.merge(DD, FN, on='VisitNumber')
        test = pd.merge(Dtest, Ftest, on='VisitNumber')

    df = pd.merge(df1, df, on='VisitNumber')
    test = pd.merge(test1, test, on='VisitNumber')

    return df, test

df = pd.read_csv("../data/train_v1.csv")
test = pd.read_csv("test.csv")

df, test = editor(df, test, type='DD')

df.to_csv('edited_train.csv', index=False)
test.to_csv('edited_test.csv', index=False)
