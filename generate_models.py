import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import logging
import joblib
import datetime

from mlflow import MlflowClient
import mlflow
from sklearn.model_selection import train_test_split
from generate_models_functions import random_forest_model, reduc_dim_lasso, print_results

# Tracking ML Flow - Model A : base price estimation
client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

def generate_models(suffix):

    logging.basicConfig(filename='./logs/logs_calculate_models_' + suffix + '.log', level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    # Tracking ML Flow - Settings
    datetime_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_A_experiment = mlflow.set_experiment(f"model_A_{suffix}_{datetime_now}")

    try:
        df = pd.read_parquet(f'databases/temp/{suffix}_05_export_dvf_completed_final.parquet')
    except:
        print(f"Dataframe with the suffix {suffix} was not found in databases/temp. Run prepare_data before calculating model.")
    
    logging.info(f'Prepare dataframe for model generation | Remove NA & useless columns | Shape before: {df.shape}')

    ##### Columns to keep
    
    # Version heavy
    """df = df[['Valeur fonciere', 'Paris', 'Type local', 'Surface reelle bati', 'Nombre pieces principales', 
    'Surface terrain', 'littoral', 'loyer', 'nb_habitants', 'revenu',
       'aeroport', 'alimentation', 'baginade', 'banques', 'boulangerie',
       'camping', 'cinema', 'college_lycee', 'creche', 'ecole', 'ecole_sup',
       'etab_hospi', 'gare', 'gare_nationale', 'hotels', 'info_touristique',
       'parcours_sportif', 'police', 'port_plaisance', 'residence_u',
       'restaurants', 'resto_u', 'salle_multisport', 'anciennete_vente', 'prix_m2_commune', 'prix_m2_gps']]"""
    
    # Version light
    df = df[['Valeur fonciere', 'Paris', 'Type local', 'Surface reelle bati', 'Nombre pieces principales', 
    'Surface terrain', 'littoral', 'loyer', 'nb_habitants', 'revenu',
       'college_lycee', 'creche', 'ecole', 'ecole_sup', 'gare', 'hotels', 'info_touristique', 'police', 'restaurants', 'salle_multisport', 
       'anciennete_vente', 'prix_m2_commune', 'prix_m2_gps']]
    
    df = df.dropna()
    logging.info(f'Prepare dataframe for model generation | Remove NA & useless columns | Shape after: {df.shape}')

    # Affiche la heatmap avec les valeurs
    plt.figure(figsize=(25, 15))
    matrice_correlation = df.corr()
    sns.heatmap(matrice_correlation, annot = True)
    plt.savefig(f"models/outputs/{suffix}_heatmap_correlation.png")




    df['Type local'] = df.apply(lambda x: 1 if x['Type local'] == "Maison" else 0, axis = 1)
    target = df['Valeur fonciere']

    # Test 1 : Appartements Province
    df_1 = df[(df['Type local'] == 0) & (df['Paris'] == 0)]
    target_1 = df_1['Valeur fonciere']
    df_1 = df_1.drop(columns = {'Valeur fonciere'})
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(df_1, target_1, test_size = 0.2, random_state = 1234)

    # Test 2 : Appartements Paris
    df_2 = df[(df['Type local'] == 0) & (df['Paris'] == 1)]
    target_2 = df_2['Valeur fonciere']
    df_2 = df_2.drop(columns = {'Valeur fonciere'})
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(df_2, target_2, test_size = 0.2, random_state = 1234)

    # Test 3 : Maisons hors Paris
    df_3 = df[(df['Type local'] == 1) & (df['Paris'] == 0)]
    target_3 = df_3['Valeur fonciere']
    df_3 = df_3.drop(columns = {'Valeur fonciere'})
    X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(df_3, target_3, test_size = 0.2, random_state = 1234)


    # Model général : tous biens
    df = df.drop(columns = {'Valeur fonciere'})
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size = 0.2, random_state = 1234)
    column_list = df.columns.values


    # Affichage des prix médians des biens
    logging.info(f'Modèle global | Median value: {target.median()}')
    logging.info(f'Modèle 1 - Appartements Province | Median value: {target_1.median()}')
    logging.info(f'Modèle 2 - Appartement Paris | Median value: {target_2.median()}')
    logging.info(f'Modèle 3 - Maisons Province | Median value: {target_3.median()}')


    # Modèle global
    logging.info(f'Generating model | Tous biens')
    rf_all, results_all = random_forest_model(X_train, X_test, y_train, y_test, suffix, "tous_biens", datetime_now)
    reduc_dim_lasso(df, X_train, X_test, y_train, y_test, column_list, "global", suffix)

    # Modèle 1 - Appartements Province
    logging.info(f'Generating model | Modèle 1 | Appartements hors Paris')
    rf_1, results_1 = random_forest_model(X_train_1, X_test_1, y_train_1, y_test_1, suffix, "appartements_province", datetime_now)
    reduc_dim_lasso(df_1, X_train_1, X_test_1, y_train_1, y_test_1, column_list, "appart_province", suffix)

    # Modèle 2 - Appartements Paris
    logging.info(f'Generating model | Modèle 2 | Appartements Paris')
    rf_2, results_2 = random_forest_model(X_train_2, X_test_2, y_train_2, y_test_2, suffix, "appartements_paris", datetime_now)
    reduc_dim_lasso(df_2, X_train_2, X_test_2, y_train_2, y_test_2, column_list, "appart_paris", suffix)

    # Modèle 3 - Maisons Province
    logging.info(f'Generating model | Modèle 3 | Maisons Province')
    rf_3, results_3 = random_forest_model(X_train_3, X_test_3, y_train_3, y_test_3, suffix, "maisons_province", datetime_now)
    reduc_dim_lasso(df_3, X_train_3, X_test_3, y_train_3, y_test_3, column_list, "maison_province", suffix)

    print_results("Modèle global", results_all)
    print_results("Modèle 1 - Appartements Province", results_1)
    print_results("Modèle 2 - Appartements Paris", results_2)
    print_results("Modèle 3 - Maisons Province", results_3)

    # Sauvegarde des modèles
    joblib.dump(rf_all, f'models/{suffix}_model_rf_all.pkl', compress = 3)
    joblib.dump(rf_1, f'models/{suffix}_model_rf_1_appart_province.pkl', compress = 3)
    joblib.dump(rf_2, f'models/{suffix}_model_rf_2_appart_paris.pkl', compress = 3)
    joblib.dump(rf_3, f'models/{suffix}_model_rf_3_maison_province.pkl', compress = 3)

    # Export de la liste des colonnes de X_train
    feature_list = list(X_train.columns)
    print(X_train.dtypes)
    df_feat = pd.DataFrame({'feature_name': feature_list})
    df_feat.to_csv(f'models/outputs/{suffix}_features_list_for_model_A.csv')