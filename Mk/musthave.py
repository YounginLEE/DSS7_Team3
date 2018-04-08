import pandas as pd

df = pd.read_csv('../data/train_v1.csv')

df['refund'] = df['ScanCount'].apply(lambda x: -x if x < 0 else 0)
df['purchase'] = df['ScanCount'].apply(lambda x: x if x > 0 else 0)

todo = ['FinelineNumber', 'DepartmentDescription', 'company_code', 'product_code']

for Todo in todo:

    TD_all = df[[Todo, 'purchase']].groupby([Todo]).mean()
    TD_all.reset_index(inplace=True)
    TD_all = TD_all.sort_values(by='purchase', ascending=False)
    TD_all = TD_all[TD_all['purchase'] >= 1]

    anyone = TD_all[Todo].unique()
    df_filtered = df[~df[Todo].isin(anyone)] # 모든 고객이 1개 이상씩 구매한 Todo 제거

    trip=df['TripType'].unique()
    trip.sort()
    base = df[['VisitNumber', 'FinelineNumber']].groupby('VisitNumber').mean().reset_index()
    base.drop('FinelineNumber' ,axis=1, inplace=True)

    for i in trip:

        TD = df_filtered[['TripType', Todo, 'purchase']].groupby(['TripType', Todo]).mean()
        TD.reset_index(inplace=True)
        # 해당 트립타입 순차적으로 투입
        TD_ = TD[TD['TripType']==i].sort_values(by=['TripType', 'purchase'], ascending=False)

        TD_ = TD_[TD_['purchase'] >= 1.0] # `평균적으로` 1개이상 산 물건들의 Todo

        TD_ = df[df[Todo].isin(TD_[Todo])]
        # TripType별로 1개 이상 구매한 제품을 몇개 샀는지 / 안산애들은 나중에 0으로 fillna
        TD_ = TD_[['VisitNumber', 'purchase']].groupby(['VisitNumber']).sum()


        TD_.reset_index(inplace=True)
        TD_.rename(columns={'purchase':'TripType_{}'.format(i)}, inplace=True)

        base = pd.merge(base, TD_,  on='VisitNumber', how='outer').fillna(0)

    base.to_csv('Musthave_by_{}.csv'.format(Todo), index=False)
