import numpy as np
import pandas as pd
from sklearn.preprocessing import *

def weekday(x):
    if x == 'Monday': return 1
    elif x == 'Tuesday' : return 2
    elif x == 'Wednesday' : return 3
    elif x == 'Thursday' : return 4
    elif x == 'Friday' : return 5
    elif x == 'Saturday' : return 6
    elif x == 'Sunday' : return 7

def FN(df, test): # 메모리 문제로 돌리지를 못함
    df.dropna(how='all', inplace=True, subset=['FinelineNumber'])

    # df['FinelineNumber'].fillna(0, inplace=True)
    # test['FinelineNumber'].fillna(0, inplace=True)
    df['FinelineNumber'].dropna(how='any', inplace=True)
    test['FinelineNumber'].dropna(how='any', inplace=True)

    df = df[df['FinelineNumber'].isin(a&b)] # train과 test 둘다 있는 것만으로 모델링
    df.reset_index(drop=True, inplace=True)
    a = set(test['FinelineNumber'].unique())
    b = set(df['FinelineNumber'].unique())

    index = test[test['FinelineNumber'].isin(a-b)].index
    test['FinelineNumber'].loc[index] = 0 # train에 없는 애들은 0 으로 설정

    lb = LabelBinarizer() # 객체 생성
    D_FL = lb.fit_transform(df['FinelineNumber'].values.tolist())
    T_FL = lb.transform(test['FinelineNumber'].values.tolist())
    T_FL.drop(0, axis=1, inplace=True) # 0으로 설정된 컬럼 삭제 (train에 없음)

    # df = pd.concat([df, pd.DataFrame(D_FL)], axis=1)
    # test = pd.concat([test, pd.DataFrame(T_FL)], axis=1)

    return D_FL, T_FL

def DD(df, test):
    df.dropna(how='all', inplace=True, subset=['DepartmentDescription'])

    a = set(test['DepartmentDescription'].unique())
    b = set(df['DepartmentDescription'].unique())
    df = df[df['DepartmentDescription'].isin(a&b)] # train과 test 둘다 있는 것만으로 모델링
    df.reset_index(drop=True, inplace=True)

    lb = LabelBinarizer()
    D_DD = lb.fit_transform(df['DepartmentDescription'].values.tolist())
    T_DD = lb.transform(test['DepartmentDescription'].values.tolist())

    df = pd.concat([df['VisitNumber'], pd.DataFrame(D_DD)], axis=1)
    test = pd.concat([test['VisitNumber'], pd.DataFrame(T_DD)], axis=1)

    df = df.groupby('VisitNumber').sum()
    df.reset_index(inplace=True)
    test = test.groupby('VisitNumber').sum()
    test.reset_index(inplace=True)

    return df, test


def editor(df, test):
    df['weekday'] = df['Weekday'].apply(weekday)
    test['weekday'] = test['Weekday'].apply(weekday)

    df['refund'] = df['ScanCount'].apply(lambda x: -x if x < 0 else 0)
    df['purchase'] = df['ScanCount'].apply(lambda x: x if x > 0 else 0)
    test['refund'] = test['ScanCount'].apply(lambda x: -x if x < 0 else 0)
    test['purchase'] = test['ScanCount'].apply(lambda x: x if x > 0 else 0)

    df1 = df[['VisitNumber', 'TripType','weekday']].groupby('VisitNumber').mean()
    df2 = df[['VisitNumber','refund','purchase']].groupby('VisitNumber').sum()
    test1 = test[['VisitNumber', 'weekday']].groupby('VisitNumber').mean()
    test2 = test[['VisitNumber','refund','purchase']].groupby('VisitNumber').sum()
    df1.reset_index(inplace=True)
    df2.reset_index(inplace=True)
    test1.reset_index(inplace=True)
    test2.reset_index(inplace=True)

    df1 = pd.merge(df1, df2, on='VisitNumber')
    test1 = pd.merge(test1, test2, on='VisitNumber')

    #bring DepartmentDescription
    df, test = DD(df, test)

    df = pd.merge(df1, df, on='VisitNumber')
    test = pd.merge(test1, test, on='VisitNumber')

    df['weekday'] = df['weekday'].astype('category')  # 카테고리화
    test['weekday'] = test['weekday'].astype('category')

    # df.dropna(how='any', inplace=True)
    # df.reset_index(drop=True, inplace=True)

    return df, test

df = pd.read_csv("../data/train_v1.csv")
test = pd.read_csv("test.csv")
df, test = editor(df, test)
df.to_csv('edited_train.csv', index=False)
test.to_csv('edited_test.csv', index=False)
