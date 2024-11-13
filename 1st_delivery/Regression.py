import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, LeaveOneOut, train_test_split, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import pandas as pd
# Set warnings to ignore
warnings.filterwarnings('ignore')

class Regression:

    def logisticRegression(self, X_train, Y_train, X_test, Y_test):
        # Scale the features for both train and test sets
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize logistic regression model with a max iteration parameter
        lr = LogisticRegression(max_iter=2000)
        lr.fit(X_train_scaled, Y_train)  # Fit model on training data
        
        # Predict on test data
        Y_pred = lr.predict(X_test_scaled)
        
        # Calculate accuracy
        acc = accuracy_score(Y_test, Y_pred)
        print("Logistic Regression model accuracy on test data (in %):", acc * 100)
        print(f"Classification Report:\n{classification_report(Y_test, Y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(Y_test, Y_pred)}")
        return acc * 100
        
    def logisticRegressionOptimized(self, X_train, Y_train, X_test, Y_test):
        # Define a pipeline to scale data and apply logistic regression
        pipe = Pipeline([('scaler', StandardScaler()), ('regressor', LogisticRegression(max_iter=1000))])
        
        # Set up grid search to tune 'C' parameter for regularization strength
        mod = GridSearchCV(pipe, param_grid={'regressor__C': [0.01, 0.1, 1, 10, 100]}, cv=5)
        mod.fit(X_train, Y_train)
        
        print("Best parameters are: ", mod.best_params_)  # Display best parameters found
        print("Best score is: ", mod.best_score_)  # Display best score
        
        # Use the best model to predict on test data
        best_model = mod.best_estimator_
        Y_pred = best_model.predict(X_test)
        acc = accuracy_score(Y_test, Y_pred)
        print("Logistic Regression model accuracy on test data (in %):", acc * 100)
        print(f"Classification Report:\n{classification_report(Y_test, Y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(Y_test, Y_pred)}")
        return acc * 100  # Return accuracy percentage
        
    def linearDiscriminantAnalysis(self, X_train, Y_train, X_test, Y_test):
        # Initialize Linear Discriminant Analysis (LDA) model
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, Y_train)  # Fit LDA model on training data
        
        # Predict on test data
        Y_pred = lda.predict(X_test)
        acc = accuracy_score(Y_test, Y_pred)
        print("Linear Discriminant Analysis model accuracy on test data (in %):", acc * 100)
        print(f"Classification Report:\n{classification_report(Y_test, Y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(Y_test, Y_pred)}")
        return acc * 100
        
    
    def linearDiscriminantAnalysisOptimized(self, X_train, Y_train, X_test, Y_test):
        # Define pipeline to scale data and apply LDA
        pipe = Pipeline([('scaler', StandardScaler()), ('regressor', LinearDiscriminantAnalysis())])
        
        # Define parameter grid for optimization, considering solver and shrinkage parameters
        param_grid = [
            {'regressor__solver': ['svd'], 'regressor__shrinkage': [None]},
            {'regressor__solver': ['lsqr', 'eigen'], 'regressor__shrinkage': [None, 'auto', 0.1, 0.5, 1.0]}
        ]
        
        # Perform grid search with 5-fold cross-validation
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
        # Initialize Quadratic Discriminant Analysis (QDA) model
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(X_train, Y_train)
        
        # Predict on test data
        Y_pred = qda.predict(X_test)
        acc = accuracy_score(Y_test, Y_pred)
        print("Quadratic Discriminant Analysis model accuracy on test data (in %):", acc * 100)
        print(f"Classification Report:\n{classification_report(Y_test, Y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(Y_test, Y_pred)}")
        return acc * 100
    
    def quadraticDiscriminantAnalysisOptimized(self, X_train, Y_train, X_test, Y_test):
        # Define pipeline to scale data and apply QDA
        pipe = Pipeline([('scaler', StandardScaler()), ('regressor', QuadraticDiscriminantAnalysis())])
        
        # Set parameter grid for QDA, tuning regularization and tolerance
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
        # Scale entire feature set
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        logisctic=LogisticRegression(max_iter=2000)
        y_pred = cross_val_predict(logisctic, X_scaled, Y, cv=cv)

        # Perform cross-validation with logistic regression
        scores = cross_val_score(logisctic, X_scaled, Y, cv=cv, scoring='accuracy')
        mean_scores = np.mean(scores) * 100
        print(f"Logistic Regression model accuracy with {cv}-fold cross-validation (in %):", mean_scores)
        print(f"Classification Report:\n{classification_report(Y, y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(Y, y_pred)}")
       
        return mean_scores
    
    def crossValidation_linearDiscriminantAnalysis(self, X, Y, cv):
        # Scale entire feature set
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lda=LinearDiscriminantAnalysis()
        y_pred = cross_val_predict(lda, X_scaled, Y, cv=cv)  

        # Perform cross-validation with LDA
        scores = cross_val_score(lda, X_scaled, Y, cv=cv, scoring='accuracy')
        mean_scores = np.mean(scores) * 100
        print(f"Linear Discriminant Analysis model accuracy with {cv}-fold cross-validation (in %):", mean_scores)
        print(f"Classification Report:\n{classification_report(Y, y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(Y, y_pred)}")
        return mean_scores
    
    def crossValidation_quadraticDiscriminantAnalysis(self, X, Y, cv):
        # Scale entire feature set
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        qda=QuadraticDiscriminantAnalysis()
        y_pred = cross_val_predict(qda, X_scaled, Y, cv=cv)  

        # Perform cross-validation with QDA
        scores = cross_val_score(QuadraticDiscriminantAnalysis(), X_scaled, Y, cv=cv, scoring='accuracy')
        mean_scores = np.mean(scores) * 100
        print(f"Quadratic Discriminant Analysis model accuracy with {cv}-fold cross-validation (in %):", mean_scores)
        print(f"Classification Report:\n{classification_report(Y, y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(Y, y_pred)}")
        return mean_scores

    def leaveOneOutCrossValidation_logisticRegression(self, X, Y):
        # Scale entire feature set
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply Leave-One-Out cross-validation with logistic regression
        loo = LeaveOneOut()
        logisctic=LogisticRegression(max_iter=2000)
        y_pred = cross_val_predict(logisctic, X_scaled, Y, cv=loo)

        scores = cross_val_score(logisctic, X_scaled, Y, cv=loo, scoring='accuracy')
        mean_scores = np.mean(scores) * 100
        print(f"Logistic Regression model accuracy with leave-one-out cross-validation (in %):", mean_scores)
        print(f"Classification Report:\n{classification_report(Y, y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(Y, y_pred)}")
       
        return mean_scores
        
    def leaveOneOutCrossValidation_linearDiscriminantAnalysis(self, X, Y):
        # Scale entire feature set
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply Leave-One-Out cross-validation with LDA
        loo = LeaveOneOut()
        lda=LinearDiscriminantAnalysis()
        y_pred = cross_val_predict(lda, X_scaled, Y, cv=loo)

        scores = cross_val_score(lda, X_scaled, Y, cv=loo, scoring='accuracy')
        mean_scores = np.mean(scores) * 100
        print(f"Linear Discriminant Analysis model accuracy with leave-one-out cross-validation (in %):", mean_scores)
        print(f"Classification Report:\n{classification_report(Y, y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(Y, y_pred)}")
       
        return mean_scores
        
    def leaveOneOutCrossValidation_quadraticDiscriminantAnalysis(self, X, Y):
        # Scale entire feature set
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply Leave-One-Out cross-validation with QDA
        loo = LeaveOneOut()
        qda=QuadraticDiscriminantAnalysis()
        y_pred = cross_val_predict(qda, X_scaled, Y, cv=loo)

        scores = cross_val_score(qda, X_scaled, Y, cv=loo, scoring='accuracy')
        mean_scores = np.mean(scores) * 100
        print(f"Quadratic Discriminant Analysis model accuracy with leave-one-out cross-validation (in %):", mean_scores)
        print(f"Classification Report:\n{classification_report(Y, y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(Y, y_pred)}")
       
        return mean_scores
        
    def bootstrap_logisticRegression(self, X_train, Y_train, X_test, Y_test, n):
        scores = []
        
        # Scale train and test sets
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train logistic regression model
        lr = LogisticRegression(max_iter=2000)
        lr.fit(X_train_scaled, Y_train)


        y_pred = lr.predict(X_test)
        
        
        # Perform bootstrapping with logistic regression
        for _ in range(n):
            X_bs, y_bs = resample(X_train_scaled, Y_train)
            lr.fit(X_bs, y_bs)
            score = lr.score(X_test_scaled, Y_test)
            scores.append(score)
        mean_score = np.mean(scores) * 100
        print(f"Bootstrap Mean Accuracy: {mean_score:.2f}%")
        print(f"Classification Report:\n", classification_report(Y_test, y_pred))
        print(f"Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))
       
        return mean_score
    
    def bootstrap_linearDiscriminantAnalysis(self, X_test, Y_test, X_train, Y_train, n):
        scores = []
        
        # Scale train and test sets
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train LDA model
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train_scaled, Y_train)
        
        y_pred = lda.predict(X_test)

        # Perform bootstrapping with LDA
        for _ in range(n):
            X_bs, y_bs = resample(X_train_scaled, Y_train)
            lda.fit(X_bs, y_bs)
            score = lda.score(X_test_scaled, Y_test)
            scores.append(score)
        
        mean_score = np.mean(scores) * 100
        print(f"Bootstrap Mean Accuracy: {mean_score:.2f}%")
        print(f"Classification Report:\n{classification_report(Y_test, y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(Y_test, y_pred)}")
     
        return mean_score
        
    def bootstrap_quadraticDiscriminantAnalysis(self, X_test, Y_test, X_train, Y_train, n):
        scores = []
        
        # Scale train and test sets
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train QDA model
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(X_train_scaled, Y_train)
        y_pred = qda.predict(X_test)

        # Perform bootstrapping with QDA
        for _ in range(n):
            X_bs, y_bs = resample(X_train_scaled, Y_train)
            qda.fit(X_bs, y_bs)
            score = qda.score(X_test_scaled, Y_test)
            scores.append(score)
        mean_score = np.mean(scores) * 100
        print(f"Classification Report:\n{classification_report(Y_test, y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(Y_test, y_pred)}")      
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