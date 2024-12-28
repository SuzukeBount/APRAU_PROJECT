import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, LeaveOneOut, train_test_split, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.utils import resample
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import random
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error




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





        #Parte 2 Trabalho
    def dt_function(self, X, Y, abc=0):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
        
        if abc==0:
             dt = DecisionTreeClassifier(random_state=42)
        else:

            dt = DecisionTreeClassifier(
                    ccp_alpha=0.0,
                    criterion='entropy',
                    max_depth=50,
                    max_leaf_nodes=100,
                    min_samples_leaf=1,
                    min_samples_split=2,
                    splitter='best',
                    random_state=42
                )
            
        dt.fit(X_train, y_train)

        y_pred = dt.predict(X_test)
        y_pred_train=dt.predict(X_train)
        acc=accuracy_score(y_test, y_pred)
        train_acc=accuracy_score(y_train, y_pred_train)

        print("Accuracy:", acc*100, "\nTrain Accuracy:", train_acc*100)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("\nConfusion Matrix", confusion_matrix(y_test, y_pred))


        # Compute the confusion matrix for classes 0, 1, 2
        cmatrix_test = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=[0, 1, 2])

        # Calculate the error rate
        # Sum of off-diagonal elements divided by total number of samples
        error_rate = (cmatrix_test.sum() - np.trace(cmatrix_test)) / cmatrix_test.sum()

        print("Error Rate:", error_rate)

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        
        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        
        # Calculate Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

        #Accuracy Score
        print(accuracy_score(y_test,y_pred))

        # Log Loss
        y_pred_proba = dt.predict_proba(X_test)
        logloss = log_loss(y_test, y_pred_proba)
        print(f"Log Loss: {logloss:.4f}")

        #ROC_AUC
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
        print(f"ROC AUC: {roc_auc:.4f}")

        disp = ConfusionMatrixDisplay(confusion_matrix=cmatrix_test, display_labels=['0', '1', '2'])
        disp.plot()

        return acc*100, dt
    

    def grid_search_decision_tree(self, X, Y):
     
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

        param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 50, 100],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_leaf_nodes': [None, 10, 50, 100],
        'ccp_alpha': [0.0, 0.01, 0.1],
        }

        # Initialize the classifier
        dt_classifier = DecisionTreeClassifier()

        # Set up the GridSearchCV
        grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, 
                                scoring='accuracy', cv=3, n_jobs=-1, verbose=1)

        # Fit the grid search to the training data
        grid_search.fit(X_train, y_train)

        # Get the best model and parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Evaluate the model on the test set
        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print("Best Parameters:", grid_search.best_params_)

        return best_model, best_params, test_accuracy
    
    def rf_function(self, X, Y, abc=0):
       
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
        if abc==0:
            rf = RandomForestClassifier(random_state=42)
        else:
            rf = RandomForestClassifier(class_weight=None, criterion='entropy', max_depth=50, max_features=None, max_leaf_nodes=100, min_samples_leaf=2, min_samples_split=2, n_estimators=100, random_state=42, verbose=0)

        rf.fit(X_train, y_train)

        # Make predictions
        y_pred = rf.predict(X_test)
        y_pred_train = rf.predict(X_train)

        # Calculate accuracies
        acc = accuracy_score(y_test, y_pred)
        train_acc = accuracy_score(y_train, y_pred_train)

        # Print results
        print("Accuracy:", acc * 100, "\nTrain Accuracy:", train_acc * 100)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

        
        # Compute the confusion matrix for classes 0, 1, 2
        cmatrix_test = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=[0, 1, 2])

             # Log Loss
        y_pred_proba = rf.predict_proba(X_test)
        logloss = log_loss(y_test, y_pred_proba)
        print(f"Log Loss: {logloss:.4f}")

        #ROC_AUC
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
        print(f"ROC AUC: {roc_auc:.4f}")
              #Accuracy Score
        print(accuracy_score(y_test,y_pred))

        # Calculate the error rate
        # Sum of off-diagonal elements divided by total number of samples
        error_rate = (cmatrix_test.sum() - np.trace(cmatrix_test)) / cmatrix_test.sum()
        print("Error Rate:", error_rate)
        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        
        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        
        # Calculate Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

  

        accuracy_score(y_test,y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cmatrix_test, display_labels=['0', '1', '2'])
        disp.plot()

        
            # Plot the first tree in the Random Forest
        plt.figure(figsize=(60, 40))
        plot_tree(rf.estimators_[0], filled=True, feature_names=X.columns, class_names=['0', '1', '2'], fontsize=6)
        plt.show()
        # Check the shapes after the split
        print("Shape of X_train:", X_train.shape)
        print("Shape of X_test:", X_test.shape)

    
        
        return acc * 100, rf


    def grid_search_random_forest(self, X, Y):
   
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

        # Define the parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 50, 100],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [2, 4],
            'max_features': ['sqrt', 'log2', None],
            'random_state': [42],
            'verbose': [0, 1],
            'class_weight': [None, 'balanced'],
            'max_leaf_nodes':[None, 10, 50, 100]
        }

        # Initialize the classifier
        rf_classifier = RandomForestClassifier()

        # Set up the GridSearchCV
        grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, 
                                scoring='accuracy', cv=3, n_jobs=-1, verbose=1)

        # Fit the grid search to the training data
        grid_search.fit(X_train, y_train)

        # Get the best model and parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Evaluate the model on the test set
        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print("Best Parameters:", grid_search.best_params_)


        return best_model, best_params, test_accuracy
    
    def build_svm_classifier(self, X, Y , kernel='linear'):

    
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=420)
        
        # Initialize SVM with the specified kernel
        svm = SVC(kernel=kernel, random_state=42)
        
        # Train the SVM classifier
        svm.fit(X_train, y_train)
        


        
        # Predict on the test set
        y_pred = svm.predict(X_test)
        
        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy with {kernel} kernel: {acc:.4f}\n")
        
        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        
        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        
        # Calculate Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

        print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
  




    def gridSearchSVM(self, X, Y, kernel):
               
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        
        # Initialize SVM with the specified kernel
        svm = SVC(kernel=kernel, random_state=42)

        # Define parameter grid for GridSearchCV
        param_grid = {
            'C': [0.1, 1],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto', 0.01],
            'coef0': [0.0, 0.1, 1.0],
            'shrinking': [True, False],
            'probability': [True, False],
            'tol': [0.001, 0.1],
            'cache_size': [200, 500],
            'class_weight': [None, 'balanced'],
            'verbose': [False],
            'max_iter': [2000],
            'decision_function_shape': ['ovr', 'ovo'],
            'break_ties': [True, False]
        }
        
        # Perform GridSearchCV to find the best parameters
        grid_search = GridSearchCV(svm, param_grid, cv=3, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Get the best estimator
        best_svm = grid_search.best_estimator_
        print("Best Parameters:", grid_search.best_params_)

    def build_svm_classifier_complete(self, X, Y , kernel ,C , degree, gamma, coef0, shrinking, prob, tol, cache, weight, verb, max_iter, decision_f_shape, breakt):


        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=420)
        
        # Initialize SVM with the specified kernel
        svm = SVC(kernel=kernel ,C=C , degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking, probability=prob, tol=tol, cache_size=cache, class_weight=weight, verbose=verb,max_iter=max_iter, decision_function_shape=decision_f_shape, break_ties=breakt, random_state=42)
        
        # Train the SVM classifier
        svm.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = svm.predict(X_test)
        
        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy with {kernel} kernel: {acc:.4f}\n")
        
        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        
        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        
        # Calculate Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")    
                

        print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
  
        return svm





class FeatureSelectionEnvironment:
    def __init__(self, X, y, max_features=5):
        self.X = X  # Input features
        self.y = y  # Target labels
        self.max_features = max_features  # Maximum number of features to select
        self.n_features = X.shape[1]  # Total number of features
        self.current_state = np.zeros(self.n_features)  # Initial state (no features selected)

    def reset(self):
        self.current_state = np.zeros(self.n_features)
        return self.current_state

    def get_action_space(self):
        # The action space consists of unselected features
        return [i for i in range(self.n_features) if self.current_state[i] == 0]

    def step(self, action):
        # Update the state (mark the feature as selected)
        self.current_state[action] = 1
        selected_features = np.where(self.current_state == 1)[0]

        # Limit the number of features selected
        if len(selected_features) > self.max_features:
            return self.current_state, -10, True  # Penalty for too many features

        # Train models and evaluate accuracy
        X_selected = self.X.iloc[:, selected_features]

        # Split data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X_selected, self.y, test_size=0.2, random_state=42)

        # Train Random Forest Classifier
        rf = RandomForestClassifier(class_weight=None, criterion='entropy', max_depth=50, max_features=None, 
                max_leaf_nodes=100, min_samples_leaf=2, min_samples_split=2, n_estimators=100, random_state=42, verbose=0)

        rf.fit(X_train, y_train)

        rf_pred = rf.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)

        # Train SVM
        svm = SVC(kernel='rbf',C=1,degree=2,gamma='scale',coef0=0.0,shrinking=True,probability=True,
                  tol=0.01,cache_size=200,class_weight=None,verbose=True,max_iter=2000,
                  decision_function_shape='ovr',break_ties=False,random_state=42)
        svm.fit(X_train, y_train)

        svm_pred = svm.predict(X_test)
        svm_accuracy = accuracy_score(y_test, svm_pred)

        # Choose the best model based on accuracy
        accuracy = max(rf_accuracy, svm_accuracy)

        # Reward: Positive for improvement, negative for no improvement or overfitting
        if accuracy > 0.7:  # Example threshold for good performance
            reward = 10
        else:
            reward = -5

        # Check if the task is done
        done = len(selected_features) >= self.max_features  # End after selecting a set of features

        return self.current_state, reward, done


# Q-Learning Agent Class
class QLearningAgent:
    def __init__(self, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1, n_features=10):
        # Initialize Q-table with zeros. The Q-table will have shape (2^n_features, n_actions)
        self.q_table = np.zeros((2**n_features, n_actions))  # 2^n_features possible states (binary representations)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration factor
        self.n_features = n_features  # Number of features

    def state_to_index(self, state):
        # Convert binary state (array of 0s and 1s) to an integer index
        state = state.astype(int)  # Convert to integers (0 or 1)
        return int("".join(state.astype(str)), 2)

    def select_action(self, available_actions, state_index):
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: Choose a random action
            return random.choice(available_actions)
        else:
            # Exploitation: Choose the best action based on Q-values
            return available_actions[np.argmax(self.q_table[state_index, available_actions])]

    def update_q_table(self, state_index, action, reward, next_state_index, done):
        best_next_action = np.argmax(self.q_table[next_state_index]) if not done else 0
        self.q_table[state_index, action] += self.alpha * (reward + self.gamma * self.q_table[next_state_index, best_next_action] - self.q_table[state_index, action])


# Function to train the Q-learning agent for feature selection
def train_q_learning_agent(X, y, max_features=5, episodes=100):
    env = FeatureSelectionEnvironment(X, y, max_features)
    agent = QLearningAgent(n_actions=X.shape[1], n_features=X.shape[1])

    rewards = []

    for episode in range(episodes):
        state = env.reset()
        state_index = agent.state_to_index(state)  # Convert initial state to an index
        done = False
        total_reward = 0
        
        while not done:
            available_actions = env.get_action_space()
            action = agent.select_action(available_actions, state_index)
            next_state, reward, done = env.step(action)
            next_state_index = agent.state_to_index(next_state)  # Convert next state to an index
            agent.update_q_table(state_index, action, reward, next_state_index, done)
            state = next_state
            state_index = next_state_index
            total_reward += reward

        rewards.append(total_reward)
        print(f"Episode {episode+1}/{episodes} - Total Reward: {total_reward}")

    # Plot the rewards over episodes to see the learning process
    plt.plot(range(1, episodes+1), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning Reward Progress')
    plt.show()

