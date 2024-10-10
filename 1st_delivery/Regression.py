from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import DataClass as dc
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

class Regression:
    def logisticRegression(self, X_train, Y_train, X_test, Y_test):
        
        pipe = Pipeline([('scaler', StandardScaler()), ('regressor', LogisticRegression(max_iter=1000))])
        mod = GridSearchCV(pipe, param_grid={'regressor__C': [0.01, 0.1, 1, 10, 100]}, cv=5)
        mod.fit(X_train, Y_train)
        
        print("Best parameters are: ", mod.best_params_)
        print("Best score is: ", mod.best_score_)
            
        best_model = mod.best_estimator_
        Y_pred = best_model.predict(X_test)
        acc = accuracy_score(Y_test, Y_pred)
        print("Logistic Regression model accuracy on test data (in %):", acc * 100)