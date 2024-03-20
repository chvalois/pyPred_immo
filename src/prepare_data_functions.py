import pandas as pd
import numpy as np
from glob import glob
import os
import logging
import unidecode
import requests, zipfile, io, shutil
import math

import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point

pd.set_option('display.max_rows', None)



### --- Lecture du ou des fichiers DVF --- ###

def escapeNan(row):
# Fonction de remplacement des NaN par 0
    val = str(row)
    if val == "nan":
        val = " "
    else:
        val = val.replace('.0','') # if from float values
        val = val + " "
    return val


def aggregate_dvf(year):
# Aggrégation des fichiers DVF
# all = True : tous les fichiers dans le répertoire databases/dvf/ seront concaténés
    if (year == "all") | (year == "All"):
        file_list = glob("databases/dvf/*.txt", recursive = False)
        logging.info("Generate DVF Dataframe | File list: " + str(file_list))

        df_full = pd.DataFrame()

        for file in file_list:
            df = pd.read_csv(file, sep = "|", decimal = ",", dtype = {'Code departement': str,
                                                                                     'Code commune': str,
                                                                                     'No Volume': str,
                                                                                     '1er lot': str,
                                                                                     '2eme lot': str, 
                                                                                     '3eme lot': str, 
                                                                                     '4eme lot': str, 
                                                                                     '5eme lot': str,
                                                                                     'Nature culture speciale': str})
            df_full = pd.concat([df_full, df]).reset_index()
            logging.info("Generate DVF Dataframe | Size of dataframe: " + str(df_full.shape))

    else:
        filename = 'databases/dvf/valeursfoncieres-' + year + '.txt'
        df_full = pd.read_csv(filename, sep = "|", decimal = ",", dtype = {'Code departement': str,
                                                                                     'Code commune': str,
                                                                                     'No Volume': str,
                                                                                     '1er lot': str,
                                                                                     '2eme lot': str, 
                                                                                     '3eme lot': str, 
                                                                                     '4eme lot': str, 
                                                                                     '5eme lot': str,
                                                                                     'Nature culture speciale': str})

    df_full = df_full.drop(columns = {'Identifiant de document', 'Reference document', '1 Articles CGI',
       '2 Articles CGI', '3 Articles CGI', '4 Articles CGI', '5 Articles CGI', 'No Volume', '1er lot',
       'Surface Carrez du 1er lot', '2eme lot', 'Surface Carrez du 2eme lot', '3eme lot', 'Surface Carrez du 3eme lot', '4eme lot',
       'Surface Carrez du 4eme lot', '5eme lot', 'Surface Carrez du 5eme lot', 'Nature culture speciale'})


    return df_full


### --- Manipulations initiales sur le fichier DVF --- ###

def transform_dvf(df):

    # Manipulations sur les codes départements et codes communes
    df['Code postal'] = df['Code postal'].fillna(-1).astype(int).astype(str).replace('-1', np.nan)
    df['Code departement'] = df['Code departement'].apply(lambda x: str(x).zfill(2) if len(str(x)) == 1 else x)
    df['Code commune'] = df['Code commune'].apply(lambda x: str(x).zfill(3) if len(str(x)) < 3 else x)
    df['Code commune INSEE'] = df['Code departement'].astype(str) + df['Code commune'].astype(str)

    # Création d'un champ "Ville" pour interroger les coordonnées GPS lorsque le champ Adresse ne donne rien
    df['Ville'] = df['Code postal'] + " " + df['Commune']

    # Modification des communes avec arrondissements
    df['Commune'] = df['Commune'].replace(['MARSEILLE 1ER', 'MARSEILLE 2EME', 'MARSEILLE 3EME', 'MARSEILLE 4EME',
                                            'MARSEILLE 5EME', 'MARSEILLE 6EME', 'MARSEILLE 7EME', 'MARSEILLE 8EME',
                                            'MARSEILLE 9EME', 'MARSEILLE 10EME', 'MARSEILLE 11EME', 'MARSEILLE 12EME',
                                            'MARSEILLE 13EME', 'MARSEILLE 14EME', 'MARSEILLE 15EME', 'MARSEILLE 16EME',
                                            'LYON 1ER', 'LYON 2EME', 'LYON 3EME', 'LYON 4EME', 'LYON 5EME',
                                            'LYON 6EME', 'LYON 7EME', 'LYON 8EME', 'LYON 9EME',
                                            'PARIS 01', 'PARIS 02', 'PARIS 03', 'PARIS 04', 'PARIS 05', 'PARIS 06',
                                            'PARIS 07', 'PARIS 08', 'PARIS 09', 'PARIS 10', 'PARIS 11', 'PARIS 12',
                                            'PARIS 13', 'PARIS 14', 'PARIS 15', 'PARIS 16', 'PARIS 17', 'PARIS 18',
                                            'PARIS 19', 'PARIS 20'],
                                            ['MARSEILLE', 'MARSEILLE', 'MARSEILLE', 'MARSEILLE',
                                            'MARSEILLE', 'MARSEILLE', 'MARSEILLE', 'MARSEILLE',
                                            'MARSEILLE', 'MARSEILLE', 'MARSEILLE', 'MARSEILLE',
                                            'MARSEILLE', 'MARSEILLE', 'MARSEILLE', 'MARSEILLE',
                                            'LYON', 'LYON', 'LYON', 'LYON', 'LYON',
                                            'LYON', 'LYON', 'LYON', 'LYON',
                                            'PARIS', 'PARIS', 'PARIS', 'PARIS', 'PARIS', 'PARIS',
                                            'PARIS', 'PARIS', 'PARIS', 'PARIS', 'PARIS', 'PARIS',
                                            'PARIS', 'PARIS', 'PARIS', 'PARIS', 'PARIS', 'PARIS',
                                            'PARIS', 'PARIS'])

    # Ajout d'apostrophes après L et D pour une meilleure reconnaissance des noms de rues 
    # et un meilleur matching lorsqu'on cherche à ajouter des données
    df['Voie'] = df['Voie'].replace([' L ', ' D '], [' L\'', ' D\''], regex = True)

    # Suppression des tirets des noms de ville
    df['Ville'] = df['Ville'].replace('-', ' ', regex=True).astype(str)

    # Création du champ "addressName" qui sera utilisée dans la requête API pour récupérer les coordonnées GPS
    df['Adresse'] = df.apply(lambda row: 
        escapeNan(row['No voie'])
        + escapeNan(row['Type de voie'])         
        + escapeNan(row['Voie'])
        + escapeNan(row['Code postal']).zfill(5) # pour respecter le format des codes postaux
        + str(row['Commune'])
        , axis = 1).str.strip()
    
    return df


### --- Extract addresses --- ###
def extract_addresses(df):

    # Extract addresses from DVF
    ad = df[['No voie', 'Type de voie', 'Voie', 'Code postal', 'Commune', 'Code departement', 'Code commune INSEE', 'Ville', 'Adresse']]

    # Removes duplicates of addresses
    ad = ad.drop_duplicates()
    ad['Code departement'] = ad['Code departement'].astype(str)

    return ad

### --- Filter DVF Database --- ###
def filter_dvf(df):

    logging.info(f"Final Clean Dataframe | Drop duplicates | Shape before: {df.shape}")
    df = df.drop_duplicates(subset = ['Date mutation', 'Valeur fonciere', 'Adresse'])
    logging.info(f"Final Clean Dataframe | Drop duplicates | Shape after: {df.shape}")

    ## Reads external additional databases
    dep_reg = pd.read_csv("databases/departements-region.csv",sep=",")

    # Keeps only sale type = "Appartement" or "Maison" et and only mutation type = "Vente"
    df = df[(df["Type local"] == "Appartement") | (df["Type local"] == "Maison")]
    df = df[df["Nature mutation"] == "Vente"]

    # Removes duplicates
    logging.info("Filtering DVF | Nb rows before removing duplicates:" + str(df.shape[0]))
    df = df.drop_duplicates(subset = None)
    df = df.drop_duplicates(subset = ["Date mutation","Valeur fonciere","Code postal"], keep = False)
    df['Surface terrain'] = df['Surface terrain'].fillna(0)
    logging.info("Filtering DVF | Nb rows after removing duplicates:" + str(df.shape[0]))

    # Calculates price / m² for each sale
    df["Prix m2"] = df["Valeur fonciere"] / df["Surface reelle bati"]
    df["Paris"]= df["Commune"].apply(lambda x : 1 if x == "PARIS" else 0)

    Paris = df[df["Paris"] == 1]
    Autres_villes = df[df["Paris"] == 0]

    # Before removing outliers
    logging.info("Filtering DVF | Nb rows Paris before removing outliers:" + str(Paris.shape[0]))

    # Handling outliers in Paris (more restrictive on Paris because more extreme values)
    outliers_max_Paris = Paris["Valeur fonciere"].quantile(0.99)
    outliers_min_Paris = Paris["Valeur fonciere"].quantile(0.05)
    outliers_max_Paris_surface = Paris["Surface reelle bati"].quantile(0.99)
    outliers_min_Paris_surface = Paris["Surface reelle bati"].quantile(0.01)
    outliers_max_Paris_prix_m2= Paris["Prix m2"].quantile(0.99)
    outliers_min_Paris_prix_m2 = Paris["Prix m2"].quantile(0.1)

    Paris = Paris[(Paris["Valeur fonciere"] < outliers_max_Paris) & 
                (Paris["Valeur fonciere"] > outliers_min_Paris) &
                (Paris["Surface reelle bati"] < outliers_max_Paris_surface) &
                (Paris["Surface reelle bati"] > outliers_min_Paris_surface) &
                (Paris["Prix m2"] < outliers_max_Paris_prix_m2) &
                (Paris["Prix m2"] > outliers_min_Paris_prix_m2) ]

    logging.info("Filtering DVF | Nb rows Paris after removing outliers: " + str(Paris.shape[0]))

    logging.info(f"Filtering DVF | Seuil des Outliers Paris | Price Min: {str(int(outliers_min_Paris))}")
    logging.info(f"Filtering DVF | Seuil des Outliers Paris | Price Max: {str(int(outliers_max_Paris))}")
    logging.info(f"Filtering DVF | Seuil des Outliers Paris | Surface Min: {str(int(outliers_min_Paris_surface))}")
    logging.info(f"Filtering DVF | Seuil des Outliers Paris | Surface Max: {str(int(outliers_max_Paris_surface))}")
    logging.info(f"Filtering DVF | Seuil des Outliers Paris | Prix m2 Min: {str(int(outliers_min_Paris_prix_m2))}")
    logging.info(f"Filtering DVF | Seuil des Outliers Paris | Prix m2 Max: {str(int(outliers_max_Paris_prix_m2))}")

    # logging.info(f"Filtering DVF | Seuil des Outliers Paris : Price Min, Price Max, Surface Min, Surface Max, Prix m² Min, Prix m² Max")
    # logging.info("Filtering DVF | " + str(int(outliers_min_Paris)) + " | " + str(int(outliers_max_Paris)) + " | " + str(int(outliers_min_Paris_surface)) + " | " + str(int(outliers_max_Paris_surface)) + " | " + str(int(outliers_min_Paris_prix_m2)) + " | " + str(int(outliers_max_Paris_prix_m2)))

    del outliers_max_Paris, outliers_min_Paris, outliers_max_Paris_surface, outliers_min_Paris_surface
    del outliers_max_Paris_prix_m2, outliers_min_Paris_prix_m2


    # Handling outliers in Autres Villes (different than PARIS)
    logging.info("Filtering DVF | Nb rows Autres Villes before removing outliers:" + str(Autres_villes.shape[0]))

    outliers_max_Autres_villes = Autres_villes["Valeur fonciere"].quantile(0.99)
    outliers_min_Autres_villes = Autres_villes["Valeur fonciere"].quantile(0.05)
    outliers_max_Autres_villes_surface = Autres_villes["Surface reelle bati"].quantile(0.99)
    outliers_min_Autres_villes_surface = Autres_villes["Surface reelle bati"].quantile(0.01)
    outliers_max_Autres_villes_prix_m2 = Autres_villes["Prix m2"].quantile(0.99)
    outliers_min_Autres_villes_prix_m2 = Autres_villes["Prix m2"].quantile(0.1)
    outliers_max_Autres_villes_surface_terrain = Autres_villes["Surface terrain"].quantile(0.99)


    Autres_villes = Autres_villes[(Autres_villes["Valeur fonciere"] < outliers_max_Autres_villes) & 
                (Autres_villes["Valeur fonciere"] > outliers_min_Autres_villes) &
                (Autres_villes["Surface reelle bati"] < outliers_max_Autres_villes_surface) &
                (Autres_villes["Surface reelle bati"] > outliers_min_Autres_villes_surface) &
                (Autres_villes["Prix m2"] < outliers_max_Autres_villes_prix_m2) &
                (Autres_villes["Prix m2"] > outliers_min_Autres_villes_prix_m2) &
                (Autres_villes["Surface terrain"] < outliers_max_Autres_villes_surface_terrain)]

    logging.info("Filtering DVF | Nb rows Autres Villes after removing outliers: " + str(Autres_villes.shape[0]))


    logging.info(f"Filtering DVF | Seuil des Outliers Province | Price Min: {str(int(outliers_min_Autres_villes))}")
    logging.info(f"Filtering DVF | Seuil des Outliers Province | Price Max: {str(int(outliers_max_Autres_villes))}")
    logging.info(f"Filtering DVF | Seuil des Outliers Province | Surface Min: {str(int(outliers_min_Autres_villes_surface))}")
    logging.info(f"Filtering DVF | Seuil des Outliers Province | Surface Max: {str(int(outliers_max_Autres_villes_surface))}")
    logging.info(f"Filtering DVF | Seuil des Outliers Province | Prix m2 Min: {str(int(outliers_min_Autres_villes_prix_m2))}")
    logging.info(f"Filtering DVF | Seuil des Outliers Province | Prix m2 Max: {str(int(outliers_max_Autres_villes_prix_m2))}")

    #logging.info("Filtering DVF | Seuil des Outliers Province : Price Min, Price Max, Surface Min, Surface Max, Prix m² Min, Prix m² Max, Surface Terrain Max")
    #logging.info("Filtering DVF | " + str(int(outliers_min_Autres_villes)) + " | " + str(int(outliers_max_Autres_villes)) + " | " + str(int(outliers_min_Autres_villes_surface)) + " | " + str(int(outliers_max_Autres_villes_surface)) + " | " + str(int(outliers_min_Autres_villes_prix_m2)) + " | " + str(int(outliers_max_Autres_villes_prix_m2)) + " | " + str(int(outliers_max_Autres_villes_surface_terrain)))

    del outliers_max_Autres_villes, outliers_min_Autres_villes, outliers_max_Autres_villes_surface
    del outliers_min_Autres_villes_surface, outliers_max_Autres_villes_prix_m2, outliers_min_Autres_villes_prix_m2
    del outliers_max_Autres_villes_surface_terrain

    df_clean = pd.concat([Paris, Autres_villes], axis = 0)
    del Paris, Autres_villes

    ## Add regions with code departement, Removes department of Corsica
    dep_reg = dep_reg.rename(columns = {"num_dep":"Code departement"})
    dico2 = {"Code departement":{"2A":"201","2B":"202"}}
    dep_reg = dep_reg.replace(dico2)
    dep_reg["Code departement"] = dep_reg["Code departement"].astype('int32')
    df_clean = df_clean.replace(dico2)
    df_clean["Code departement"] = df_clean["Code departement"].astype('int32')
    df_clean = df_clean.merge(right = dep_reg , on = 'Code departement', how = 'left')
    df_clean = df_clean[(df_clean["Code departement"] != 201) & (df_clean["Code departement"] != 202)]

    del dep_reg, dico2

    return df_clean

# ------ Traitement des adresses ------ #

def abbr_type_voie(adresse, abbreviations):
    words = adresse.split()
    for i, word in enumerate(words):
        if word.lower() in abbreviations:
            words[i] = abbreviations[word.lower()]
    return ' '.join(words)


abbreviations = {
    'allée': 'ALL',
    'avenue': 'AV',
    'boulevard': 'BD',
    'carrefour': 'CAR',
    'carré': 'CARR',
    'centre': 'CTR',
    'chalet': 'CHT',
    'chemin': 'CH',
    'cité': 'CTE',
    'clos': 'CLS',
    'corniche': 'COR',
    'cour': 'CR',
    'cours': 'CRS',
    'domaine': 'DOM',
    'esplanade': 'ESP',
    'galerie': 'GAL',
    'grand\'place': 'GPE',
    'grande rue': 'GR',
    'hameau': 'HAM',
    'haut': 'HT',
    'impasse': 'IMP',
    'jardin': 'JDN',
    'lieu dit': 'LD',
    'lotissement': 'LOT',
    'montée': 'MT',
    'passage': 'PAS',
    'petite rue': 'PTR',
    'place': 'PL',
    'placis': 'PLC',
    'port': 'POR',
    'quai': 'QUAI',
    'rond point': 'RPT',
    'route': 'RTE',
    'rue': 'RUE',
    'sente': 'SEN',
    'sentier': 'SEN',
    'square': 'SQ',
    'terrain': 'TER',
    'traverse': 'TRA',
    'vallée': 'VAL',
    'venelle': 'VEN',
    'vieux chemin': 'VCH',
    'villa': 'VLA',
    'village': 'VLG',
    'voie': 'VOIE',
    'général': 'GEN'
}


def add_gps_coord_to_df(df_dvf, df_ad):
    # Transform the dataframe "Address" in DVF format
    df_ad['NUMBER'] = df_ad['NUMBER'].astype('int')
    df_ad['POSTCODE'] = df_ad['POSTCODE'].astype('int')
    df_ad['STREET'] = df_ad['STREET'].apply(lambda x: abbr_type_voie(x, abbreviations))
    df_ad['STREET'] = df_ad['STREET'].apply(lambda x: unidecode.unidecode(x.upper()))
    df_ad['CITY'] = df_ad['CITY'].apply(lambda x: unidecode.unidecode(x.upper()))
    df_ad['Adresse'] = df_ad.apply(lambda x: str(x['NUMBER']) + " " + x['STREET'] + " " + str(x['POSTCODE']) + " " + x['CITY'], axis = 1)
    
    # Calculates a dataframe containing the mean of GPS coords for a given street 
    # (enables to fill with these coordinates when no exact match with street number)
    ad_mean_gps = df_ad.groupby(['STREET', 'CITY']).agg({'LON' : 'mean', 'LAT': 'mean'}).reset_index().rename(columns = {'LON': 'LON_mean', 'LAT': 'LAT_mean'})

    # Removes NA from DVF Dataframe (à voir)
    logging.info("Filtering DVF | Nb Rows before removing NA from DVF Dataframe:" + str(df_dvf.shape[0]))
    # df_dvf = df_dvf.dropna()
    logging.info("Filtering DVF | Nb Rows after removing NA from DVF Dataframe:" + str(df_dvf.shape[0]))

    # Add GPS coordinates to DVF dataframe
    # Step 1 - merge exact GPS coordinates if match with street number
    dfll = df_dvf.merge(df_ad[['Adresse', 'LON', 'LAT']], on = "Adresse", how = "left")
    dfll['Rue'] = dfll['Type de voie'] + " " + dfll['Voie']
    # Step 2 - merge mean street GPS coordinates if no exact match with street number
    dfll = dfll.merge(ad_mean_gps, left_on = ['Rue', 'Commune'], right_on = ['STREET', 'CITY'])
    dfll.loc[dfll['LON'].isna(), 'LON'] = dfll['LON_mean']
    dfll.loc[dfll['LAT'].isna(), 'LAT'] = dfll['LAT_mean']
    dfll = dfll.drop(columns = {'Rue', 'STREET', 'CITY', 'LON_mean', 'LAT_mean'})
    # Step 3 - removes rows not in Metropolitan France
    logging.info(f"Filtering DVF | Drop GPS KO | Shape before: {dfll.shape}")
    dfll[(dfll['LON'].between(-5.0, 8.5)) & (dfll['LAT'].between(42.0, 51.5))].shape  
    logging.info(f"Filtering DVF | Drop GPS KO | Shape after: {dfll.shape}")  
    
    logging.info("Filtering DVF | Nb Rows after adding GPS coordinates with coordinates NA:" + str(dfll['LON'].isna().sum()))
    return dfll


##### -------- Traitement des IRIS Codes -------- #####

def add_iris_code(df):
    df['dep'] = df['Code departement'].apply(lambda x: "0" + str(x) if len(str(x)) == 1 else str(x))

    # Load the shapefile
    areas = gpd.read_file('databases/CONTOURS-IRIS_2-1__SHP__FRA_2022-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2022-06-00180/CONTOURS-IRIS_2-1_SHP_LAMB93_FXX-2022/CONTOURS-IRIS.shp')
    areas = areas.to_crs('4326')
    areas['dep'] = areas['INSEE_COM'].apply(lambda x: x[:2])
    areas.head()

    df['code_iris'] = '00000000'
    df['commune_iris'] = ''
    df['nom_iris'] = ''

    dep_list = sorted(list(set(df['dep'])))
    new_df = pd.DataFrame()

    for dep in dep_list:
        df_dep = df[df['dep'] == dep]
        areas_dep = areas[areas['dep'] == dep]
        
        df_dep = add_iris_to_df(df_dep, areas_dep)
        new_df = pd.concat([new_df, df_dep])

    return new_df


def get_iris(areas, longitude, latitude):

    location = Point(longitude, latitude)
    polygon = areas.contains(location)
    
    code_iris = str(areas.loc[polygon, 'CODE_IRIS'].values[0])
    commune_iris = areas.loc[polygon, 'NOM_COM'].values[0]
    nom_iris = areas.loc[polygon, 'NOM_IRIS'].values[0]

    return code_iris, commune_iris, nom_iris

def add_iris_to_df(df_dep, areas_dep):

    for i, row in df_dep.iterrows():
        longitude = row['LON']
        latitude = row['LAT']
        try:
            code_iris, commune_iris, nom_iris = get_iris(areas_dep, longitude, latitude)

            df_dep.loc[i, 'code_iris'] = code_iris
            df_dep.loc[i, 'commune_iris'] = commune_iris
            df_dep.loc[i, 'nom_iris'] = nom_iris
        except:
            print('Erreur index', i)

    return df_dep




##### -------- DPE (Diagnostics de Performance Energetique) -------- #####

def reduce_dpe(dpe_file):

       dpe = pd.read_csv(f'databases/dpe/{dpe_file}', low_memory=False)

       dpe = dpe[['consommation_energie',
       'classe_consommation_energie', 'estimation_ges',
       'classe_estimation_ges', 'annee_construction', 'surface_habitable',
       'commune', 'numero_rue', 'type_voie', 'nom_rue', 'etage', 'code_postal',
       'code_insee_commune_actualise', 'surface_baies_orientees_nord',
       'surface_baies_orientees_est_ouest', 'surface_baies_orientees_sud', 'commune', 'arrondissement', 'type_voie', 'nom_rue', 'numero_rue',
       'code_postal', 'code_insee_commune', 'code_insee_commune_actualise']]

       return dpe

def get_dpe_files_produce_df():
# Get dpe files for each French Department and concatenate them in one CSV file

    for i in range(1, 96):

        try:
            r = requests.get(f"https://files.data.gouv.fr/ademe/dpe-departements/{i}.zip")
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall("databases/dpe/")

            file_list = os.listdir(f'databases/dpe/{i}')
            file_list.remove('td001_dpe-clean.csv')

            for file in file_list:
                os.remove(f'databases/dpe/{i}/{file}')

            shutil.move(f'databases/dpe/{i}/td001_dpe-clean.csv', f'databases/dpe/td001_dpe-clean_{1}.csv')
        except:
            pass

    dpe_df = pd.DataFrame()

    dpe_list = os.listdir('databases/dpe')

    for file in dpe_list:
        dpe = reduce_dpe(file)
        dpe_df = pd.concat([dpe_df, dpe])

    dpe_df.to_csv('dpe_france.csv')
    return dpe_df


def merge_with_dpe(df, dpe):
# Add DPE information to dataframe

    # Traitement des adresses de la base initiale pour le match avec la base DPE
    df['Adresse'] = df['Adresse'].replace(',', ' ', regex=True).astype(str)
    df['Adresse'] = df['Adresse'].replace('-', ' ', regex=True).astype(str)
    df['Adresse'] = df['Adresse'].apply(lambda x: unidecode.unidecode(x))
    df['Adresse'] = df['Adresse'].replace([' L ', ' D '], [' L\'', ' D\''], regex = True)
    df['Adresse'].head()

    # Traitement du code département, car va être important pour merger dépt par dépt avec la base DPE
    df['Code departement'] = df['Code departement'].apply(lambda x: str(x).zfill(2))
    
    # On ne conserve que les colonnes pertinentes
    dpe = dpe[['dep', 'Adresse', 'anciennete', 'consommation_energie', 'classe_consommation_energie',
                'estimation_ges', 'classe_estimation_ges', 'surface_baies_orientees_sud']]

    # On ne conserve que la première ligne trouvée (cela peut être encore amélioré)
    dpe = dpe.drop_duplicates(subset = ['Adresse'], keep='first')  
    
    # Faire en sorte que le code département soit toujours à deux chiffres
    dpe['dep'] = dpe['dep'].apply(lambda x: str(x).zfill(2))

    # Merge de la base initiale avec la base DPE sur la base de l'adresse : boucle qui va tourner pour chaque département

    merge = pd.DataFrame()

    for i in range(1,96):
        try:
            i = str(i).zfill(2)
            df_temp = df[df['Code departement'] == i]
            dpe_temp = dpe[dpe['dep'] == i]
            merge_temp = df_temp.merge(dpe_temp, on = "Adresse", how = "left")
            merge = pd.concat([merge, merge_temp])
        except:
            pass

    print(df.shape)
    print(merge.shape)
    print(merge.isna().sum())

    # Nombre de lignes de la base initiale pour laquelle aucune correspondance DPE n'a été trouvée
    nb_lines_ko = merge[merge['anciennete'].isna()].shape[0]
    print('Il restera ' + str(merge.shape[0] - nb_lines_ko) + ' lignes')


##### -------- Cleaning Dataframe -------- #####

def clean_df(df):
    df = df[['Date mutation', 'Nature mutation', 'Valeur fonciere',
         'Code postal', 'Commune', 'Code departement', 'Code commune', 'Code commune INSEE',
         'Adresse', 'Paris', 'dep_name', 'region_name',
         'Nombre de lots', 'Code type local', 'Type local', 
         'Surface reelle bati', 'Nombre pieces principales', 'Surface terrain', 
         'Prix m2', 'LON', 'LAT', 'dep', 'code_iris', 'commune_iris', 'nom_iris']]
    
    logging.info(f"Clean Dataframe | clean code department, code commune, code iris | Shape before: {df.shape}")
    df = df[df['Code departement'] <= 95]
    df['code_insee_5'] = df['Code commune INSEE'].apply(lambda x: str(x).zfill(5))
    df = df[df['code_iris'] != 0]
    df['code_iris'] = df['code_iris'].apply(lambda x: str(x).zfill(9))
    df['code_iris'] = df['code_iris'].astype(str)
    logging.info(f"Clean Dataframe | clean code department, code commune, code iris | Shape after: {df.shape}")


    # On ajoute Surface Terrain = 0 lorsque NaN (la plupart du temps, il s'agit d'appartements qui n'ont par définition pas de terrain)
    df['Surface terrain'] = df['Surface terrain'].fillna(0)

    return df


##### -------- Additional dataframes -------- #####

def add_littoral(df):
# Ajout de la feature littoral pour identifier les communes littorales
    
    mer = pd.read_csv('databases/communes_littorales.csv', sep = ";", encoding='unicode_escape')

    df = df.merge(mer, on = "Commune", how = "left")
    df['littoral'] = df['Classement'].apply(lambda x: 1 if x == "Mer" else 0)
    df = df.drop(columns = {'Classement'})

    logging.info(f"Extend dataframe | Add communes littorales - Done")

    return df

def add_loyers(df):
# Ajout des loyers moyens
    
    # Loyers Appartements 2022
    loy_app = pd.read_csv('databases/loyers/pred-app-mef-dhup.csv', sep = ";", encoding='unicode_escape')
    loy_app = loy_app[['INSEE_C', 'loypredm2']].rename(columns = {'INSEE_C': 'code_insee_5', 'loypredm2': 'loyer_appart'})
    loy_app['loyer_appart'] = loy_app['loyer_appart'].str.replace(',', '.').astype(np.float16)

    # Loyers Maisons 2022
    loy_maison = pd.read_csv('databases/loyers/pred-mai-mef-dhup.csv', sep = ";", encoding='unicode_escape')
    loy_maison = loy_maison[['INSEE_C', 'loypredm2']].rename(columns = {'INSEE_C': 'code_insee_5', 'loypredm2': 'loyer_maison'})
    loy_maison['loyer_maison'] = loy_maison['loyer_maison'].str.replace(',', '.').astype(np.float16)

    df = df.merge(loy_app, on = "code_insee_5", how = "left")
    df = df.merge(loy_maison, on = "code_insee_5", how = "left")

    df['loyer'] = df.apply(lambda x: x['loyer_maison'] if x['Type local'] == "Maison" else x['loyer_appart'], axis = 1)
    df = df.drop(columns = {'loyer_appart', 'loyer_maison'})

    logging.info(f"Extend dataframe | Add loyers - Done")


    return df

def add_population(df):
# Ajout du nombre d'habitants par commune

    # Population 2020
    pop = pd.read_csv('databases/population/base-cc-evol-struct-pop-2020.csv', sep = ";", dtype = {'CODGEO': 'str'})
    pop = pop[['CODGEO', 'P20_POP']].rename(columns = {'CODGEO': 'code_insee_5', 'P20_POP': 'nb_habitants'})

    df = df.merge(pop, on = "code_insee_5", how = "left")

    # On remplace le nb_habitants NA par 2000 qui correspond à la taille approx. d'une petite commune
    # Hypothèse : les communes qui n'ont pas de correspondance dans la base population sont petites
    df['nb_habitants'] = df['nb_habitants'].fillna(2000)

    logging.info(f"Extend dataframe | Add population - Done")

    return df

def add_revenus(df):
# Ajout des revenus médians par code IRIS ou par commune

    # Revenus iris 2020
    rev_iris = pd.read_csv('databases/revenu/BASE_TD_FILO_DISP_IRIS_2020.csv', sep = ";")
    rev_iris = rev_iris[['IRIS', 'DISP_MED20']].rename(columns = {'IRIS': 'code_iris', 'DISP_MED20': 'revenu_iris_2020'})
    rev_iris['revenu_iris_2020'] = rev_iris['revenu_iris_2020'].str.replace('ns', 'na').replace('nd', 'na').replace('s', 'na')
    rev_iris = rev_iris.replace('na', np.nan)

    df = df.merge(rev_iris, on = "code_iris", how = "left")

    # Revenus médians 2021
    rev = pd.read_csv('databases/revenu/cc_filosofi_2021_COM.csv', sep = ";", dtype = {'CODGEO': str})
    rev = rev[['CODGEO', 'MED21']].rename(columns = {'CODGEO': 'code_insee_5', 'MED21': 'revenu_com_2021'})
    rev['revenu_com_2021'] = rev['revenu_com_2021'].str.replace('ns', 'na').replace('nd', 'na').replace('s', 'na')
    rev = rev.replace('na', np.nan)

    df = df.merge(rev, on = "code_insee_5", how = "left")

    df['revenu_iris_2020'] = df['revenu_iris_2020'].fillna(df['revenu_iris_2020'].median())
    df['revenu_com_2021'] = df['revenu_com_2021'].fillna(df['revenu_com_2021'].median())
    df['revenu'] = df.apply(lambda x: x['revenu_com_2021'] if x['revenu_com_2021'] != 0 else x['revenu_iris_2020'], axis = 1)

    logging.info(f"Extend dataframe | Add revenus | Shape before removing revenu with NA: {df.shape}")
    df = df.dropna(subset = 'revenu')
    df['revenu'] = df['revenu'].astype(np.float64)
    logging.info(f"Extend dataframe | Add revenus | Shape after removing revenu with NA: {df.shape}")

    df = df.drop(columns = {'revenu_iris_2020', 'revenu_com_2021'})


    logging.info(f"Extend dataframe | Add revenus - Done")

    return df

# Mapping Equipements

eq_dic = {"A101" : "police",
       "A104" : "police",
       "A203" : "banques",
       "A504" : "restaurants",
       "B101" : "alimentation",
       "B102" : "alimentation",
       "B201" : "alimentation",
       "B202" : "alimentation",
       "B203" : "boulangerie",
       "C101" : "ecole",
       "C102" : "ecole",
       "C104" : "ecole",
       "C105" : "ecole",
       "C201" : "college_lycee",
       "C301" : "college_lycee",
       "C302" : "college_lycee",
       "C303" : "college_lycee",
       "C304" : "college_lycee",
       "C305" : "college_lycee",
       "C501" : "ecole_sup",
       "C502" : "ecole_sup",
       "C503" : "ecole_sup",
       "C504" : "ecole_sup",
       "C701" : "residence_u",
       "C702" : "resto_u",
       "D101" : "etab_hospi",
       "D102" : "etab_hospi",
       "D103" : "etab_hospi",
       "D106" : "etab_hospi",
       "D107" : "etab_hospi",
       "D301" : "pharmacie",
       "D502" : "creche",
       "E102" : "aeroport",
       "E107" : "gare_nationale",
       "E108" : "gare",
       "E109" : "gare",
       "F109" : "parcours_sportif",
       "F121" : "salle_multisport",
       "F201" : "baginade",
       "F202" : "port_plaisance",
       "F303" : "cinema",
       "G102" : "hotels",
       "G103" : "camping",
       "G104" : "info_touristique"       
      }

def add_bpe(df):
# Ajout du nombre d'équipements par commune

    bpe = pd.read_csv('databases/bpe/bpe21_ensemble.csv', sep = ";", dtype = {'BV2012': str,
                                                               'DEP': str,
                                                               'DEPCOM': str,
                                                               'EPCI': str})
    bpe = bpe[['DCIRIS', 'TYPEQU', 'NB_EQUIP', 'DEP']].rename(columns = {'DCIRIS': 'code_iris', 'NB_EQUIP': 'nb_equipements'})

    # Suppression des lignes correspondant à la Corse et aux DOM TOM
    bpe = bpe[-((bpe["DEP"] == "2A") | (bpe["DEP"] == "2B"))]
    bpe["DEP"] = bpe["DEP"].astype(int)
    bpe = bpe[bpe['DEP'] <= 95]
    bpe = bpe.drop(columns = {'DEP'})

    eq_to_keep = ["A101", "A104", "A203", "A504", "B101", "B102", "B201", "B202", "B203", "C101", "C102", "C104", "C105", "C201", "C301", "C302", "C304", "C305", "C501", "C502", "C503", "C504", "C701", "C702","D101", "D102", "D103", "D106", "D107", "D301","D502", "E102", "E107", "E108", "E109", "F109","F121", "F201", "F202", "F303", "G102", "G103", "G104"]
    bpe = bpe[bpe['TYPEQU'].isin(eq_to_keep)]
    bpe = bpe.replace({'TYPEQU': eq_dic})
    bpe = pd.pivot_table(bpe, values=['nb_equipements'], index=['code_iris'], columns=['TYPEQU'], aggfunc="sum", fill_value=0)
    bpe = bpe.droplevel(0, axis=1)

    df = df.merge(bpe, on = "code_iris", how = "left")
    logging.info(f"Extend dataframe | Add base permanente des equipements - Done")

    return df

##### -------- Final Cleaning Dataframe completed -------- #####

def final_clean(df, suffix):
    
    # Calcul de la date de référence : date max du dataframe
    df['Date mutation'] = pd.to_datetime(df['Date mutation'], format="%d/%m/%Y")
    df['date_ref'] = pd.to_datetime(df['Date mutation'].max(), format="%d/%m/%Y")

    # Calcul de l'anciennete de la vente basée sur la date max contenue dans le modèle
    df['anciennete_vente'] = ((df['date_ref'] - df['Date mutation']) / np.timedelta64 (1, 'M'))
    df['anciennete_vente'] = df['anciennete_vente'].apply(lambda x: math.floor(x))

    # Calcul de l'année de la vente (en cas de fusion des bases DVF)
    df['year'] = df['Date mutation'].dt.year

    # On conserve ce qu'il y a après 2020
    # df = df[df['year'] >= 2020]
    
    # Réévaluation a posteriori des prix (+3% / an)
    # df['Prix m2 init'] = df['Prix m2']
    # df['Prix m2'] = df.apply(lambda x: x['Prix m2'] * (1 + (x['anciennete_vente'] * 3 / 100 / 12)), axis = 1)

    # df['Valeur fonciere init'] = df['Valeur fonciere']
    # df['Valeur fonciere'] = df['Prix m2'] * df['Surface reelle bati']

    # Calcul du prix moyen au m² par commune, type local et nb pièces
    df_prix_commune = df.groupby(['Code postal', 'Commune', 'Type local'])['Prix m2'].median()
    df = df.merge(df_prix_commune, how = "left", on = ['Code postal', 'Commune', 'Type local']).rename(columns = {
            'Prix m2_x': 'prix_m2', 
            'Prix m2_y': 'prix_m2_commune'})

    # Calcul du prix moyen au m² par proximité GPS et type
    df['lon_2'] = df['LON'].apply(lambda x: round(x, 2))     # On arrondit la longitude à 2 décimales afin de regrouper les biens par carré de 700m x 700m environ
    df['lat_2'] = df['LAT'].apply(lambda x: round(x, 2))     # Pareil pour latitude

    df_prix_gps = df.groupby(['lon_2', 'lat_2', 'Type local'])['prix_m2'].median()
    df = df.merge(df_prix_gps, how = "left", on = ['lon_2', 'lat_2', 'Type local']).rename(columns = {
            'prix_m2_x': 'prix_m2', 
            'prix_m2_y': 'prix_m2_gps'})
    
    # Export des prix / m²
    df_prix_commune.to_csv(f'models/{suffix}_prix_m2_commune.csv')
    df_prix_gps.to_csv(f'models/{suffix}_prix_m2_gps.csv')

    logging.info(f"Final Clean Dataframe | Shape: {df.shape}")

    return df


