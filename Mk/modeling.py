from sklearn.ensemble import *
import pandas as pd

#bring edited data
X = pd.read_csv('edited_train.csv')
y = X['TripType']
X = X.drop('TripType', axis=1)

#make model
forest = RandomForestClassifier(max_depth=5, n_estimators=200)
forest.fit(X, y)

#make answer as kaggle format
test = pd.read_csv('edited_test.csv')
t_visit=test['VisitNumber']
ans = pd.DataFrame(data = forest.predict_proba(test),
                   columns=list(map(lambda x: 'TripType_{}'.format(x), forest.classes_.astype(str))))
ans = pd.concat([t_visit, ans], axis=1)

ans.to_csv('ans.csv', index=False)
