import pandas as pd

df = pd.read_csv('../data/train_v1.csv')
test = pd.read_csv('../data/test_v1.csv')

df['refund'] = df['ScanCount'].apply(lambda x: -x if x < 0 else 0)
df['purchase'] = df['ScanCount'].apply(lambda x: x if x > 0 else 0)
test['refund'] = test['ScanCount'].apply(lambda x: -x if x < 0 else 0)
test['purchase'] = test['ScanCount'].apply(lambda x: x if x > 0 else 0)


todo = ['FinelineNumber', 'DepartmentDescription', 'company_code', 'product_code']

for Todo in todo:
    #train
    TD_all = df[[Todo, 'purchase']].groupby([Todo]).mean()
    TD_all.reset_index(inplace=True)
    TD_all = TD_all.sort_values(by='purchase', ascending=False)

    #test
    test_all = test[[Todo, 'purchase']].groupby([Todo]).mean()
    test_all.reset_index(inplace=True)
    test_all = test_all.sort_values(by='purchase', ascending=False)

    # frequent item select
    if Todo == 'DepartmentDescription':
        TD_all = TD_all[TD_all['purchase'] >= 1.3]
        test_all = test_all[test_all['purchase'] >= 1.3]
    elif Todo == 'FinelineNumber':
        TD_all = TD_all[TD_all['purchase'] >= 2]
        test_all = test_all[test_all['purchase'] >= 2]
    elif Todo == 'company_code':
        TD_all = TD_all[TD_all['purchase'] >= 2]
        test_all = test_all[test_all['purchase'] >= 2]
    elif Todo == 'product_code':
        TD_all = TD_all[TD_all['purchase'] >= 2]
        test_all = test_all[test_all['purchase'] >= 2]


    #check intersection
    df_FN = set(TD_all[Todo].unique())
    test_FN = set(test_all[Todo].unique())

    # test, train이 빈번하게 구매한 Todo 제거
    anyone = df_FN|test_FN
    df_filtered = df[~df[Todo].isin(anyone)]
    test_filtered = test[~test[Todo].isin(anyone)]

    # VisitNumber 별로 붙일 베이스 생성
    base = df[['VisitNumber', 'FinelineNumber']].groupby('VisitNumber').mean().reset_index()
    base.drop('FinelineNumber' ,axis=1, inplace=True)
    testbase = test[['VisitNumber', 'FinelineNumber']].groupby('VisitNumber').mean().reset_index()
    testbase.drop('FinelineNumber' ,axis=1, inplace=True)

    trip=df['TripType'].unique()
    trip.sort()

    for i in trip:

        TD = df_filtered[['TripType', Todo, 'purchase']].groupby(['TripType', Todo]).mean()
        TD.reset_index(inplace=True)
        TD_ = TD[TD['TripType']==i].sort_values(by=['TripType', 'purchase'], ascending=False)
        TD_ = TD_[TD_['purchase'] >= 1.0] # `평균적으로` 1개이상 산 물건들의 Todo
        # trip별 평균 1개이상 산 물건들 목록 저장
        unique_item = list(TD_[Todo].unique())

        test_TD = test_filtered[[Todo, 'purchase']].groupby(Todo).mean()
        test_TD.reset_index(inplace=True)

        # TripType별로 1개 이상 구매한 제품을 몇개 샀는지 / 안산애들은 나중에 0으로 fillna
        TD_ = df[df[Todo].isin(unique_item)]
        TD_ = TD_[['VisitNumber', 'purchase']].groupby(['VisitNumber']).sum()
        TD_.reset_index(inplace=True)
        TD_.rename(columns={'purchase':'TripType_{}'.format(i)}, inplace=True)

        test_TD_ = test[test[Todo].isin(unique_item)]
        test_TD_ = test_TD_[['VisitNumber', 'purchase']].groupby(['VisitNumber']).sum()
        test_TD_.reset_index(inplace=True)
        test_TD_.rename(columns={'purchase':'TripType_{}'.format(i)}, inplace=True)


        base = pd.merge(base, TD_, on='VisitNumber', how='outer').fillna(0)
        testbase = pd.merge(testbase, test_TD_, on='VisitNumber', how='outer').fillna(0)


    base.to_csv('Musthave_by_{}.csv'.format(Todo), index=False)
    testbase.to_csv('T_Musthave_by_{}.csv'.format(Todo), index=False)
