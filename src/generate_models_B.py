import pandas as pd
import logging
import joblib
import datetime
import mlflow
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .generate_models_functions import random_forest_model, aggregate_classified_ads, process_nlp_ads, prepare_df_model_b

# Tracking ML Flow
client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

def generate_models_B(limit):
    """ Fonction qui va générer un modèle B de type Bonus / Malus à partir d'un pris de base en fonction des propriétés d'un bien immobilier
        Arguments : 
            limit: nombre d'annonces sélectionnées pour générer le modèle, 0 si toutes les annonces doivent être prises en compte 
    """
    
    logging.basicConfig(filename='logs/logs_calculate_models_B.log', level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    # Tracking ML Flow - Settings
    datetime_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_experiment = mlflow.set_experiment(f"model_B_{datetime_now}")


    ##### ----- Génération des modèles B (Bonus/Malus) ----- #####
    logging.info(f'Aggrégation des annonces scrapées')
    df_ads = aggregate_classified_ads(limit)

    logging.info(f'Analyse NLP des annonces pour identification des entités nommées')
    df_ads = process_nlp_ads(df_ads)

    logging.info(f'Préparation du dataframe pour génération du modèle')
    df_ads_appart, df_ads_maison = prepare_df_model_b(df_ads)
    
    target_a = df_ads_appart['evol_prix_m2']
    features_a = df_ads_appart.drop(columns = {'prix', 'nb_pieces', 'surface', 'surface_terrain', 'nb_chambres', 'prix_m2', 'evol_prix_m2'})

    target_m = df_ads_maison['evol_prix_m2']
    features_m = df_ads_maison.drop(columns = {'prix', 'nb_pieces', 'surface', 'surface_terrain', 'nb_chambres', 'prix_m2', 'evol_prix_m2'})

   # Paramètres du modèle B
    params_b = {
    "n_estimators": 50,
    "max_depth": 200,
    "min_samples_leaf": 25,
    "random_state": 123
    }

    # Modèle B (bonus/malus) - Maisons
    logging.info(f'Entraînement du modèle pour les maisons')
    X_train, X_test, y_train, y_test = train_test_split(features_m, target_m, test_size = 0.2, random_state = 234)
    scaler = MinMaxScaler().fit(X_train) 
    X_train = scaler.transform(X_train) 
    X_test = scaler.transform(X_test)
    rf_maison, results_maison = random_forest_model(X_train, X_test, y_train, y_test, "", "B", "maisons", datetime_now, params_b)

    # Modèle B (bonus/malus) - Appartements
    logging.info(f'Entraînement du modèle pour les appartements')
    X_train, X_test, y_train, y_test = train_test_split(features_a, target_a, test_size = 0.2, random_state = 234)
    scaler = MinMaxScaler().fit(X_train) 
    X_train = scaler.transform(X_train) 
    X_test = scaler.transform(X_test)
    rf_appart, results_appart = random_forest_model(X_train, X_test, y_train, y_test, "", "B", "appartments", datetime_now, params_b)

    # Export de la liste des features
    feature_list = list(features_m.columns)
    df_feat = pd.DataFrame({'feature_name': feature_list})
    df_feat.to_csv(f'models/features_list_for_model_B.csv')

    logging.info(f'Export des modèles')
    joblib.dump(rf_maison, f'models/model_B_features_maison.pkl', compress = 3)
    joblib.dump(rf_appart, f'models/model_B_features_appart.pkl', compress = 3)

    logging.info(f'Génération du modèle B effectuée avec succès')
