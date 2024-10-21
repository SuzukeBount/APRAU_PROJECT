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
       #print(f"Classification Report:\n{classification_report(Y_test, Y_pred)}")
        #print(f"Confusion Matrix:\n{confusion_matrix(Y_test, Y_pred)}")
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
        return acc * 100
        
        
    def linearDiscriminantAnalysis(self, X_train, Y_train, X_test, Y_test):
        
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, Y_train)
        Y_pred = lda.predict(X_test)
        acc = accuracy_score(Y_test, Y_pred)
        print("Linear Discriminant Analysis model accuracy on test data (in %):", acc * 100)
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
        return acc * 100
    
    def quadraticDiscriminantAnalysis(self, X_train, Y_train, X_test, Y_test):
        
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(X_train, Y_train)
        Y_pred = qda.predict(X_test)
        acc = accuracy_score(Y_test, Y_pred)
        print("Quadratic Discriminant Analysis model accuracy on test data (in %):", acc * 100) 
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
        print(f"Bootstrap Mean Accuracy: {mean_score:.2f}%")
        return mean_score
    


    def ridgeCrossValidation(self, X, Y, cv):
        X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=42)
        ridge_cv=RidgeCV(alphas=[0.1,1,10.0,100.0], cv=cv)
        ridge_cv.fit(X_train, Y_train)
        ridge_best_alpha=ridge_cv.alpha_
        ridge_coeficient=ridge_cv.coef_
        print(f"Best alpha for Ridge regression:",ridge_best_alpha)
        print(f"Coeficient for Ridge regression:",ridge_coeficient)
        return ridge_best_alpha, ridge_coeficient
    
    def LassoCrossValidatiln(self, X, Y, cv):
        X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=42)
        lassocv=LassoCV(alphas=[0.1,1,10.0,100.0], cv=cv)
        lassocv.fit(X_train, Y_train)
        lasso_best_alpha=lassocv.alpha_
        lasso_coeficient=lassocv.coef_
        print(f"Best alpha for Lasso regression:",lasso_best_alpha)
        print(f"Coeficient for Lasso regression:",lasso_coeficient)

    def elasticNetValidation(self, X, Y, cv):
        X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=42)
        elasticnet_cv=ElasticNetCV(alphas=[0.1,1,10.0,100.0], cv=cv)
        elasticnet_cv.fit(X_train, Y_train)
        print(f"Best alpha for ElasticNet regression:",elasticnet_cv.alpha_)
        print(f"awdawdawd", elasticnet_cv.alphas_)
        print(f"Coeficient for ElasticNet regression:",elasticnet_cv.coef_)



    def gridSearchRidge(self, X, Y, cv):
        X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=42)

        ridge=Ridge()
        param_grid = {'alpha': np.logspace(-3, 3, 50)} 
        grid_search=GridSearchCV(estimator=ridge, param_grid=param_grid, cv=cv)
        
        grid_search.fit(X_train, Y_train)
        best_alpha= grid_search.best_params_['alpha']
        print(f"Best alpha with Grid search is: {best_alpha}" )

    def gridSearchLasso(self, X, Y, cv):
        X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=42)
        lasso=Lasso()
        param_grid= {'alpha': np.logspace(-3,3,50)}
        grid_search= GridSearchCV(estimator=lasso, param_grid=param_grid, cv=cv)

        grid_search.fit(X_train, Y_train)
        best_alpha= grid_search.best_params_['alpha']
        print(f"Best alpha with Grid search is: {best_alpha}")

    def gridSearchElasticNet(self, X, Y, cv):
        X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=42)
        elastic_net=ElasticNet()
        param_grid= {'alpha': np.logspace(-3,3,50)}
        grid_search= GridSearchCV(estimator=elastic_net, param_grid=param_grid, cv=cv)

        grid_search.fit(X_train, Y_train)
        best_alpha= grid_search.best_params_['alpha']
        print(f"Best alpha with Grid search is: {best_alpha}")    




    def ridgeRegression(self, X, Y, alpha):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

        ridge_model = Ridge(alpha=alpha)
        ridge_model.fit(X_train, y_train)
        ridge_predictions = ridge_model.predict(X_test)

        ridge_mse = mean_squared_error(y_test, ridge_predictions)
        ridge_r2 = r2_score(y_test, ridge_predictions)


        print(f"Mean Squared Error: {ridge_mse}")
        print(f"RMSE", np.sqrt(mean_squared_error(y_test, ridge_predictions)))
        print(f"Model coefficients: ", (ridge_model.coef_))
        print(f"R^2 Score: {ridge_r2}")
        print(f"Ridge Training Score: ", ridge_model.score(X_train, y_train) * 100)
        print(f"Ridge Testing Score: ", ridge_model.score(X_test, y_test) * 100)

    def lassoRegression(self, X, Y, alpha):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

        lasso_model = Lasso(alpha=alpha)
        lasso_model.fit(X_train, y_train)
        lasso_predictions = lasso_model.predict(X_test)

        lasso_mse = mean_squared_error(y_test, lasso_predictions)
        lasso_r2 = r2_score(y_test, lasso_predictions)

        print(f"Mean Squared Error: {lasso_mse}")
        print(f"RMSE", np.sqrt(mean_squared_error(y_test, lasso_predictions)))
        print(f"Model coefficients: ", (lasso_model.coef_))
        print(f"R^2 Score: {lasso_r2}")
        print(f"Lasso Training Score: ", lasso_model.score(X_train, y_train) * 100)
        print(f"Lasso Testing Score: ", lasso_model.score(X_test, y_test) * 100)
    
    
    def linearRegression(self, X, Y):
        X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=42)
        linear_model=LinearRegression()
        linear_model.fit(X_train, Y_train)
        y_pred_linear=linear_model.predict(X_test)

        print(f"Coeficient :\n{linear_model.coef_}")
        print(f"Intercept :\n{linear_model.intercept_}")
        print(f"Mean Square Error:\n{mean_squared_error(Y_test, y_pred_linear)*100}")
        print(f"Mean Absolute Error:\n{mean_absolute_error(Y_test, y_pred_linear)*100}")
        print(f"R-Squared:\n{r2_score(Y_test, y_pred_linear)*100}")
        print("Linear Regression Model Training Score: ",linear_model.score(X_train, Y_train)*100)
        print("Linear Regression Model Testing Score: ",linear_model.score(X_test, Y_test)*100)

    def elasticNetRegression(self, X, Y, alpha, ratio):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

        elastic_net_model = ElasticNet(alpha=alpha, l1_ratio=ratio)  # l1_ratio=0.5 for equal contribution from Lasso and Ridge
        elastic_net_model.fit(X_train, y_train)
        elastic_net_predictions = elastic_net_model.predict(X_test)
    

        elastic_mse = mean_squared_error(y_test, elastic_net_predictions)
        elastic_r2 = r2_score(y_test, elastic_net_predictions)


        print(f"Mean Squared Error: {elastic_mse}")
        print(f"RMSE", np.sqrt(mean_squared_error(y_test, elastic_net_predictions)))
        print(f"Model coefficients: ", (elastic_net_model.coef_))
        print(f"R^2 Score: {elastic_r2}")
        print(f"Elastic Net Training Score: ", elastic_net_model.score(X_train, y_train) * 100)
        print(f"Elastic Net Testing Score: ", elastic_net_model.score(X_test, y_test) * 100)