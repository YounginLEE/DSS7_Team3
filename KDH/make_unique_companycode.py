import pandas as pd

train = pd.read_csv('../data/train_v1.csv')
train['FinelineNumber'] = train['FinelineNumber'].astype(int)

test = pd.read_csv('../data/test_v1.csv')
test = test.fillna(value = 9999)
test['FinelineNumber'] = test['FinelineNumber'].astype(int)

df = train[['TripType','VisitNumber', 'company_code']]
df_ = test[['VisitNumber', 'company_code']]

type_list = list(df['TripType'].unique())
type_list.sort()

def get_company_set(df, type_list):

    set_list = []

    for i in type_list:
        td = df[df['TripType'].isin([i])]
        td_fine = list(td['company_code'].unique())
        set_list.append(td_fine)

    return set_list

company_set_list = get_company_set(df, type_list)

def get_type_company_unique(set_list,type_list):

    unique_list = []

    for i in range(len(type_list)):
        t_type = type_list[i]
        tmp_unique = set_list[i]
        for j in range(len(set_list)):

            if type_list[j] != t_type:
                tmp_unique = list(set(tmp_unique) - set(set_list[j]))

        unique_list.append(tmp_unique)
    return unique_list

company_unique_list = get_type_company_unique(company_set_list, type_list)

def get_unique_com_columns(df, unique_list):

    tmp_df  = df[['VisitNumber','Weekday']].drop_duplicates()

    for i in unique_list:
        t_df = df[df['company_code'] == i]
        t_df = t_df.groupby(['VisitNumber'], as_index = False)['ScanCount'].sum()
        t_df.rename(columns={'ScanCount': 'comcode_%s' %(i)}, inplace = True)

        tmp_df = tmp_df.merge(t_df, how = 'left', on = 'VisitNumber', copy = True)
        tmp_df['comcode_%s' %(i)].fillna(value = 0, inplace = True)

    tmp_df.drop(columns=['Weekday'], inplace = True)

    return tmp_df

c_unique_list = []

for i in company_unique_list:
    for j in i:
        c_unique_list.append(j)

train_c = get_unique_com_columns(train, c_unique_list)
test_c = get_unique_com_columns(test, c_unique_list)

train_c.to_csv('../data/unique_com.csv', index = False)
test_c.to_csv('../data/t_unique_com.csv', index = False)
