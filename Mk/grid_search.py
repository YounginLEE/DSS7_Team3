from sklearn.model_selection import *
from sklearn.ensemble import *
import pandas as pd
import pickle

#bring edited data
X = pd.read_csv('edited_train.csv')
y = X['TripType']
X = df.drop('TripType', axis=1)

#make model
forest = RandomForestClassifier(n_jobs=-1,
                                max_features='sqrt',
                                n_estimators=50, oob_score = True)

param_grid = {
    'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
}

CV_rfc = GridSearchCV(estimator=forest, param_grid=param_grid, cv= 5)
CV_rfc.fit(X, y)

print(CV_rfc.best_params_)
best_param = CV_rfc.best_params_

with open("best_param.txt", "wb") as fp:   #Pickling
    pickle.dump(best_param, fp)
