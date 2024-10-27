from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import DataClass as dc
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, LeaveOneOut
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.utils import resample


class Regression:





    def logisticRegression(self, X_train, Y_train, X_test, Y_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        lr = LogisticRegression(max_iter=2000)
        lr.fit(X_train_scaled, Y_train)
        
        Y_pred = lr.predict(X_test_scaled)
        
        acc = accuracy_score(Y_test, Y_pred)
        print("Logistic Regression model accuracy on test data (in %):", acc * 100)
        print(f"Classification Report:\n{classification_report(Y_test, Y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(Y_test, Y_pred)}")
        return acc * 100
        
    def logisticRegressionOptimized(self, X_train, Y_train, X_test, Y_test):
        
        pipe = Pipeline([('scaler', StandardScaler()), ('regressor', LogisticRegression(max_iter=1000))])
        mod = GridSearchCV(pipe, param_grid={'regressor__C': [0.01, 0.1, 1, 10, 100]}, cv=5)
        mod.fit(X_train, Y_train)
        
        print("Best parameters are: ", mod.best_params_)
        print("Best score is: ", mod.best_score_)
            
        best_model = mod.best_estimator_
        Y_pred = best_model.predict(X_test)
        acc = accuracy_score(Y_test, Y_pred)
        print("Logistic Regression model accuracy on test data (in %):", acc * 100)
        print(f"Classification Report:\n{classification_report(Y_test, Y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(Y_test, Y_pred)}")
        return acc * 100
        
        
    def linearDiscriminantAnalysis(self, X_train, Y_train, X_test, Y_test):
        
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, Y_train)
        Y_pred = lda.predict(X_test)
        acc = accuracy_score(Y_test, Y_pred)
        print("Linear Discriminant Analysis model accuracy on test data (in %):", acc * 100)
        print(f"Classification Report:\n{classification_report(Y_test, Y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(Y_test, Y_pred)}")
        return acc * 100
        
    
    def linearDiscriminantAnalysisOptimized(self, X_train, Y_train, X_test, Y_test):
        pipe = Pipeline([('scaler', StandardScaler()), ('regressor', LinearDiscriminantAnalysis())])
        
        param_grid = [
            {'regressor__solver': ['svd'], 'regressor__shrinkage': [None]},
            {'regressor__solver': ['lsqr', 'eigen'], 'regressor__shrinkage': [None, 'auto', 0.1, 0.5, 1.0]}
        ]
        
        mod = GridSearchCV(pipe, param_grid=param_grid, cv=5)
        mod.fit(X_train, Y_train)
        
        print("Best parameters are: ", mod.best_params_)
        print("Best score is: ", mod.best_score_)
            
        best_model = mod.best_estimator_
        Y_pred = best_model.predict(X_test)
        acc = accuracy_score(Y_test, Y_pred)
        print("Linear Discriminant Analysis model accuracy on test data (in %):", acc * 100)
        print(f"Classification Report:\n{classification_report(Y_test, Y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(Y_test, Y_pred)}")
        return acc * 100
    
    def quadraticDiscriminantAnalysis(self, X_train, Y_train, X_test, Y_test):
        
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(X_train, Y_train)
        Y_pred = qda.predict(X_test)
        acc = accuracy_score(Y_test, Y_pred)
        print("Quadratic Discriminant Analysis model accuracy on test data (in %):", acc * 100) 
        print(f"Classification Report:\n{classification_report(Y_test, Y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(Y_test, Y_pred)}")
        return acc * 100

    
    def quadraticDiscriminantAnalysisOptimized(self, X_train, Y_train, X_test, Y_test): 
        pipe = Pipeline([('scaler', StandardScaler()), ('regressor', QuadraticDiscriminantAnalysis())])
        
        param_grid = {
            'regressor__reg_param': [0.0, 0.1, 0.5, 1.0],
            'regressor__tol': [1e-4, 1e-3, 1e-2, 1e-1]
        }
        
        mod = GridSearchCV(pipe, param_grid=param_grid, cv=5)
        mod.fit(X_train, Y_train)
        
        print("Best parameters are: ", mod.best_params_)
        print("Best score is: ", mod.best_score_)
            
        best_model = mod.best_estimator_
        Y_pred = best_model.predict(X_test)
        acc = accuracy_score(Y_test, Y_pred)
        print("Quadratic Discriminant Analysis model accuracy on test data (in %):", acc * 100)  
        print(f"Classification Report:\n{classification_report(Y_test, Y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(Y_test, Y_pred)}")
        return acc * 100
        




    def crossValidation_logisticRegression(self, X, Y, cv):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Realizar validação cruzada
        scores = cross_val_score(LogisticRegression(max_iter=2000), X_scaled, Y, cv=cv, scoring='accuracy')
        mean_scores = np.mean(scores) * 100
        print(f"Logistic Regression model accuracy with {cv}-fold cross-validation (in %):", mean_scores)
       
        return mean_scores
    
    def crossValidation_linearDiscriminantAnalysis(self, X, Y, cv):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Realizar validação cruzada
        scores = cross_val_score(LinearDiscriminantAnalysis(), X_scaled, Y, cv=cv, scoring='accuracy')
        mean_scores = np.mean(scores) * 100
        print(f"Linear Discriminant Analysis model accuracy with {cv}-fold cross-validation (in %):", mean_scores)
        
        return mean_scores
    
    def crossValidation_quadraticDiscriminantAnalysis(self, X, Y, cv):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        scores = cross_val_score(QuadraticDiscriminantAnalysis(), X_scaled, Y, cv=cv, scoring='accuracy')
        mean_scores = np.mean(scores) * 100
        print(f"Quadratic Discriminant Analysis model accuracy with {cv}-fold cross-validation (in %):", mean_scores)
 
        return mean_scores

    def leaveOneOutCrossValidation_logisticRegression(self, X, Y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        loo = LeaveOneOut()
        scores = cross_val_score(LogisticRegression(max_iter=2000), X_scaled, Y, cv=loo, scoring='accuracy')
        mean_scores = np.mean(scores) * 100
        print(f"Logistic Regression model accuracy with leave-one-out cross-validation (in %):", mean_scores)

        return mean_scores
        
    def leaveOneOutCrossValidation_linearDiscriminantAnalysis(self, X, Y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        loo = LeaveOneOut()
        scores = cross_val_score(LinearDiscriminantAnalysis(), X_scaled, Y, cv=loo, scoring='accuracy')
        mean_scores = np.mean(scores) * 100
        print(f"Linear Discriminant Analysis model accuracy with leave-one-out cross-validation (in %):", mean_scores)
        return mean_scores
        
    def leaveOneOutCrossValidation_quadraticDiscriminantAnalysis(self, X, Y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        loo = LeaveOneOut()
        scores = cross_val_score(QuadraticDiscriminantAnalysis(), X_scaled, Y, cv=loo, scoring='accuracy')
        mean_scores = np.mean(scores) * 100
        print(f"Quadratic Discriminant Analysis model accuracy with leave-one-out cross-validation (in %):", mean_scores)
        return mean_scores
        
    def bootstrap_logisticRegression(self, X_train, Y_train, n):
        scores = []
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        lr = LogisticRegression(max_iter=2000)
        lr.fit(X_train_scaled, Y_train)
        
        for _ in range(n):
            X_bs, y_bs = resample(X_train_scaled, Y_train)
            lr.fit(X_bs, y_bs)
            score = lr.score(X_train_scaled, Y_train) 
            scores.append(score)
        mean_score = np.mean(scores) * 100
        print(f"Bootstrap Mean Accuracy: {mean_score:.2f}%")
       
        return mean_score
    
    def bootstrap_linearDiscriminantAnalysis(self, X, Y, n):
        scores = []
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_scaled, Y)
        
        for _ in range(n):
            X_bs, y_bs = resample(X, Y)
            lda.fit(X_bs, y_bs)
            score = lda.score(X, Y) 
            scores.append(score)
        
        mean_score = np.mean(scores) * 100
        print(f"Bootstrap Mean Accuracy: {mean_score:.2f}%")
     
        return mean_score
        
    def bootstrap_quadraticDiscriminantAnalysis(self, X, Y, n):
        scores = []
        
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(X, Y)
        
        for _ in range(n):
            X_bs, y_bs = resample(X, Y)
            qda.fit(X_bs, y_bs)
            score = qda.score(X, Y) 
            scores.append(score)
        mean_score = np.mean(scores) * 100
       
        return mean_score
    







    def bestCforLogictic(self, X, Y, cv):
        X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=42)

        log_reg = LogisticRegression(solver='saga', max_iter=1000) 

        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

        grid_search = GridSearchCV(log_reg, param_grid, cv=cv, scoring='accuracy') 
        grid_search.fit(X_train, Y_train)

        results = pd.DataFrame(grid_search.cv_results_)

        sorted_results = results.sort_values(by='rank_test_score')
        top_10_Cs = sorted_results[['param_C', 'mean_test_score', 'rank_test_score']].head(10)
        print(top_10_Cs)




    def logisticRegressionRidgeLasso(self, X, Y, penalty, C):
        X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        lr = LogisticRegression(penalty=penalty, solver='saga', max_iter=1000, C=C)
        lr.fit(X_train_scaled, Y_train)
        
        Y_pred = lr.predict(X_test_scaled)
        
        acc = accuracy_score(Y_test, Y_pred)
        print("Logistic Regression model accuracy on test data (in %):", acc * 100)
        print(f"Classification Report:\n{classification_report(Y_test, Y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(Y_test, Y_pred)}")
        print("Coefficients:", lr.coef_)



    def logisticRegressionElastic(self, X, Y, l1, C):
        X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        lr = LogisticRegression(penalty='elasticnet', solver='saga',max_iter=2000, C=C, l1_ratio=l1)
        lr.fit(X_train_scaled, Y_train)
        
        Y_pred = lr.predict(X_test_scaled)
        
        acc = accuracy_score(Y_test, Y_pred)
        print("Logistic Regression model accuracy on test data (in %):", acc * 100)
        print(f"Classification Report:\n{classification_report(Y_test, Y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(Y_test, Y_pred)}")
        print("Coefficients:", lr.coef_)



    def logisticRegressionRegularization(self, X_train, Y_train, X_test, Y_test, penalty, C):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        lr = LogisticRegression(solver='saga', penalty=penalty, C=C, max_iter=2000)
        lr.fit(X_train_scaled, Y_train)
        
        Y_pred = lr.predict(X_test_scaled)
        
        acc = accuracy_score(Y_test, Y_pred)
        print("Logistic Regression model accuracy on test data (in %):", acc * 100)
        print(f"Classification Report:\n{classification_report(Y_test, Y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(Y_test, Y_pred)}")
        return acc * 100
    
    def crossValidation_logisticRegressionWithLasso(self, X, Y, cv, penalty, C):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Realizar validação cruzada
        scores = cross_val_score(LogisticRegression(solver='saga',penalty=penalty, C=C, max_iter=2000), X_scaled, Y, cv=cv, scoring='accuracy')
        mean_scores = np.mean(scores) * 100
        print(f"Logistic Regression model accuracy with {cv}-fold cross-validation (in %):", mean_scores)
        return mean_scores

    def leaveOneOutCrossValidation_logisticRegressionRegularization(self, X, Y, penalty, C):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        loo = LeaveOneOut()
        scores = cross_val_score(LogisticRegression(solver='saga', penalty=penalty, C=C, max_iter=2000), X_scaled, Y, cv=loo, scoring='accuracy')
        mean_scores = np.mean(scores) * 100
        print(f"Logistic Regression model accuracy with leave-one-out cross-validation (in %):", mean_scores)
        return mean_scores

    def bootstrap_logisticRegressionRegularization(self, X_train, Y_train, n, penalty, C):
        scores = []
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        lr = LogisticRegression(solver='saga', penalty=penalty, C=C, max_iter=2000)
        lr.fit(X_train_scaled, Y_train)
        
        for _ in range(n):
            X_bs, y_bs = resample(X_train_scaled, Y_train)
            lr.fit(X_bs, y_bs)
            score = lr.score(X_train_scaled, Y_train) 
            scores.append(score)
        mean_score = np.mean(scores) * 100
        print(f"Bootstrap Mean Accuracy: {mean_score:.2f}%")
        return mean_score

