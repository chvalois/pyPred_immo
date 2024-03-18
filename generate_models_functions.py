import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow

import logging


##### ---------- MACHINE LEARNING FUNCTIONS ---------- #####

# Fonction d'entraînement de modèle Random Forest

def random_forest_model(X_train, X_test, y_train, y_test, suffix, model_type, datetime_now):

    logging.info(f'Generate model | X_train.shape: {X_train.shape}')
    logging.info(f'Generate model | X_test.shape: {X_test.shape}')
    logging.info(f'Generate model | y_train.shape: {y_train.shape}')
    logging.info(f'Generate model | y_test.shape: {y_test.shape}')

    # Train model
    params = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_leaf": 5,
    "random_state": 123
    }
    
    regr = RandomForestRegressor(**params)
    regr.fit(X_train, y_train)
    
    y_pred_train = regr.predict(X_train) 
    y_pred_test = regr.predict(X_test)

    score_train = regr.score(X_train, y_train)
    score_test = regr.score(X_test, y_test)
    rmse_train = mean_squared_error(y_pred_train, y_train, squared=False)
    rmse_test = mean_squared_error(y_pred_test, y_test, squared=False)
    mae_train = mean_absolute_error(y_pred_train, y_train)
    mae_test = mean_absolute_error(y_pred_test, y_test)
    r2_test = r2_score(y_pred_test, y_test)

    metrics = {"mae": mae_test, "rmse": rmse_test, "r2": r2_test}
    
    results = {'score_train': score_train, 'score_test': score_test,
              'rmse_train': rmse_train, 'rmse_test': rmse_test,
              'mae_train': mae_train, 'mae_test': mae_test}

    logging.info(f'Generate model | score train: {score_train}')
    logging.info(f'Generate model | score test: {score_test}')
    logging.info(f'Generate model | rmse train: {rmse_train}')
    logging.info(f'Generate model | rmse test: {rmse_test}')
    logging.info(f'Generate model | mae train: {mae_train}')
    logging.info(f'Generate model | mae test: {mae_test}')
    
    # Tracking ML Flow
    run_name = f"model_A_{model_type}"
    artifact_path = f"rf_a_{suffix}_{datetime_now}_{model_type}"
    
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=regr, input_example=X_test, artifact_path=artifact_path
        )


    return regr, results



# Fonction de réduction de dimension de type lasso

from sklearn.linear_model import Lasso

def reduc_dim_lasso(df, X_train, X_test, y_train, y_test, column_list, model, suffix):
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    lasso_r = Lasso(alpha = 1)
    lasso_r.fit(X_train, y_train)

    sel = SelectFromModel(estimator = lasso_r, threshold = 1e-10)
    sel.fit(X_train, y_train)

    mask = sel.get_support()
    plt.matshow(mask.reshape(1,-1), cmap = 'gray_r')
    plt.xlabel('Axe des features')
    plt.savefig(f"models/outputs/{suffix}_{model}_lasso_reduc_dimension.png")


    # Afficher un graphique représentant la valeur estimée du coefficient pour chaque variable de data
    plt.figure(figsize = (20, 10))

    lasso_coef = lasso_r.coef_ 
    plt.bar(range(len(df.columns)), lasso_coef) 
    plt.xticks(range(len(df.columns)), column_list, rotation=70)
    plt.savefig(f"models/outputs/{suffix}_{model}_lasso_coeff_value.png")


def print_results(model_name, results):
    print(model_name + " : ")
    print("Score Test: " + str(round(results['score_test'], 4)))
    print("MAE Test: " + str(round(results['mae_test'], 0)))
    print('\n')


def search_best_params(X_train, y_train):
### Archive Recherche Meilleurs Hyperparamètres

    rf = RandomForestRegressor(max_features='sqrt')  

    param_grid = {  
            "n_estimators" : [20, 50, 100],  
            "max_depth" : [250, 500, 1000],  
            "min_samples_leaf" : [5, 15, 25]}

    CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid)  
    CV_rf.fit(X_train, y_train)  
    print(CV_rf.best_params_)