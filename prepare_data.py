import warnings
#from pandas.core.common import SettingWithCopyWarning
#warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import json
import requests
import time
import logging

import os
from glob import glob
from prepare_data_functions import aggregate_dvf, transform_dvf, extract_addresses, filter_dvf, add_gps_coord_to_df, get_dpe_files_produce_df, add_iris_code, merge_with_dpe, \
                    clean_df, add_littoral, add_loyers, add_population, add_revenus, add_bpe, final_clean

def prepare_data(year, suffix):

    # Paramétrage du fichier de logs
    logging.basicConfig(filename='./logs/logs_generate_dvf_' + suffix + '.log', level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    # Chargement du jeu de données DVF
    df = aggregate_dvf(year)

    # Préparation du dataframe (manipulations adresses) DVF
    df = transform_dvf(df)

    # Extraction des adresses des biens vendus à partir du dataframe
    ad = extract_addresses(df)
    ad.to_parquet(f'databases/temp/{suffix}_00_export_dvf_adresses.parquet')

    # Premier filtrage du dataframe (clean duplicates, etc.)
    df_clean = filter_dvf(df)

    # Export de la base qui sera enrichie des codes GPS et IRIS dans le script #2C
    df_clean.to_parquet(f'databases/temp/{suffix}_01_dvf_sans_gps_iris.parquet')

    # Ajout des coordonnées GPS à partir d'une base externe d'adresses au Dataframe enrichi DVF
    df_ad = pd.read_parquet("databases/addresses_france_sans_arrondissement.parquet")
    dfll = add_gps_coord_to_df(df_clean, df_ad)
    dfll.to_parquet(f'databases/temp/{suffix}_02_export_dvf_with_coordinates.parquet')

    # Add IRIS codes to DVF Dataframe
    dfll_with_iris = add_iris_code(dfll)
    dfll_with_iris.to_parquet(f'databases/temp/{suffix}_03_export_dvf_with_coordinates_with_iris.parquet')

    # Generate DPE file in directory database
    # dpe = get_dpe_files_produce_df()

    # Merge DF with DPE
    # dpe = pd.read_csv('databases/dpe/dpe_france.csv', low_memory = False)
    #df_completed = merge_with_dpe(dfll_with_iris, dpe)
    #print(df_completed.head())
    
    # Merge DF with additional dataframes
    df = clean_df(dfll_with_iris)
    df = add_littoral(df)
    df = add_loyers(df)
    df = add_population(df)
    df = add_revenus(df)
    df = add_bpe(df)
    df.to_parquet(f'databases/temp/{suffix}_04_export_dvf_completed.parquet')

    df = final_clean(df)
    df.to_parquet(f'databases/temp/{suffix}_05_export_dvf_completed_final.parquet')
