import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel, f_regression, mutual_info_regression, RFE, RFECV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow

import logging

import re
import spacy
from string import punctuation
import unidecode
from spacy import displacy
from spacy.tokens import Span
from spacy.matcher import PhraseMatcher



##### ---------- MACHINE LEARNING FUNCTIONS ---------- #####

# Fonction d'entraînement de modèle Random Forest - modèle A price estimation

def random_forest_model(X_train, X_test, y_train, y_test, suffix, model_type, model_subtype, datetime_now, params):

    logging.info(f'Generate model | X_train.shape: {X_train.shape}')
    logging.info(f'Generate model | X_test.shape: {X_test.shape}')
    logging.info(f'Generate model | y_train.shape: {y_train.shape}')
    logging.info(f'Generate model | y_test.shape: {y_test.shape}')
    
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
    run_name = f"model_{model_type}_{model_subtype}"
    artifact_path = f"rf_{model_type}_{suffix}_{datetime_now}_{model_subtype}"
    
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


##### ---------- FONCTIONS D'ANALYSE DES ANNONCES SCRAPEES ---------- #####


def return_code_postal(text):
    try:
        result = re.findall('[0-9]{5}',text)[0]
    except:
        result = ''
    return result

def preprocess_text(text):
    text = text.lower()  # Lowercase text
    #text = re.sub(f"[{re.escape(punctuation)}]", " ", text)  # Remove punctuation
    text = " ".join(text.split())  # Remove extra spaces, tabs, and new lines
    text = unidecode.unidecode(text)
    return text


def aggregate_classified_ads(limit):
    """ Créé un dataframe contenant les informations des annonces immobilières scrapées
        Si limit = 0, alors toutes les annonces sont prises en compte """

    # Aggrégation des annonces scrapées 
    df = pd.read_csv('databases/superimmo/superimmo_20211008_33.csv')
    df2 = pd.read_csv('databases/superimmo/superimmo_20211008_13.csv')
    df3 = pd.read_csv('databases/superimmo/superimmo_20211008_69.csv')
    df = pd.concat([df, df2, df3])
    
    # Si l'argument limit = 0, on garde toutes les annonces, sinon on conserve un échantillon de la taille indiquée dans l'argument limit
    if limit > 0:
        df = df.sample(n=limit, random_state=123)
    
    # Suppression des maisons et appartements neufs, on ne conserve que les annonces qui se rapportent à l'ancien
    df = df[(df['type'] == 'maison') | (df['type'] == 'appartement')]

    # Récupération du code postal à partir de la localisation
    df['code_postal'] = df['location'].apply(lambda x: return_code_postal(x))

    # Data processing sur le prix, surface, nb pieces et nb chambres
    df['prix_title'] = df['prix_title'].replace(['€', ' '], ['', ''], regex = True)
    df['prix'] = df['prix'].replace(['€', ' '], ['', ''], regex = True)
    df['prix'] = df.apply(lambda x: x['prix_title'] if x['prix'] == "NaN" else x['prix'], axis = 1)
    df['surface_terrain'] = df['terrain'].replace(['ter.', 'm²', ' '], ['', '', ''], regex = True)
    df['nb_pieces'] = df['piece'].replace(['pièces', 'pièce', ' '], ['', '', ''], regex = True)
    df['nb_chambres'] = df['chambre'].replace(['chambres', 'chambre', ' '], ['', '', ''], regex = True)
    df = df.drop(columns = {'prix_title', 'terrain', 'piece', 'chambre'})
    df = df.dropna(axis = 0, how = 'all', subset = ['description'])
    df['desc_clean'] = df['description'].map(preprocess_text)
    df['agence_only'] = df['agence'].apply(lambda x: x.split('Publiée')[0])
    df = df.drop_duplicates(subset = 'desc_clean').drop(columns = {'Unnamed: 0'}).reset_index()

    return df


def show_ents(doc):
    """ Affiche des infos d'entités nommées basiques """
    if doc.ents:
        for ent in doc.ents:
            print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))
    else:
        print('No named entities found.')


def add_immo_feature(feature_name, liste, nlp, matcher, feature_name_list, color_list, color_pal):
    """ Matche les annonces avec une feature """
    list_p = [nlp(item) for item in liste]
    matcher.add(feature_name, None, *list_p)

    feature_name_list.append(feature_name)
    color_list.append(color_pal[len(color_list)%20])


def ents_features(i, doc, ner_lst, df):
    if doc.ents:
        for ent in doc.ents:
            if ent.label_ not in ner_lst:
                df['features'][i].append(ent.label_)
                df[ent.label_][i] = 1
    else:
        print('No named entities found.')


def process_nlp_ads(df):
    """ Fonction qui applique une analyse NLP aux annonces pour identifier les propriétés des annonces """

    # Import du corpus Spacy
    # Use following command if missing (TODO in .sh): python -m spacy download fr_core_news_md
    nlp = spacy.load('fr_core_news_md')
    
    nlp.max_length = 5000000
    # Constitution du corpus
    n_lines = df.shape[0]
    text = ""

    for i in range(0, n_lines):
        annonce_desc = df['desc_clean'][i]
        text += annonce_desc

    doc = nlp(text)

    # Import PhraseMatcher and create a matcher object:
    matcher = PhraseMatcher(nlp.vocab)

    # Définition de la palette de couleurs pour affichage des entités nommées
    palette = "hls"
    color_pal = sns.color_palette(palette, as_cmap = False, n_colors = 20)
    color_pal = color_pal.as_hex()
    color_list = []
    feature_name_list = []
    colors = {}

    # Définition du dictionnaire de features à retrouver dans les annonces
    immo_features = {
        'Exposition_Sud': ['exposition sud', 'expose plein sud', 'exposition plein sud', 'oriente sud', 'orientation sud', 
                        'plein sud', 'expose sud', 'exposition sud ouest', 'exposition sud-ouest', 'exposition sud est',
                        'exposition sud-est'],
        
        'Lumineux': ['ensoleillement', 'ensoleille', 'lumineux', 'lumineuse', 'clair', 'claire', 'lumiere', 'luminosite', 'soleil'],
        'Calme': ['calme', 'tranquille', 'preserve', 'preservee', 'double vitrage', 'silencieux'],
        'Volume': ['hauteur sous plafond', 'beau volume', 'beaux volumes', 'grands volumes', 'spacieux', 'vaste'],
        'Standing': ['standing', 'luxe', 'luxueux', 'luxueuse', 'moulures', 'dorures', 'pierre de taille', 'prestige'],
        'Charme': ['poutre', 'poutres', 'colombage', 'colombages', 'charme', 'charmant', 'coup de coeur', 'cheminee'],
        'Mezzanine': ['mezzanine', 'mezanine'],
        'Avec_Ascenseur': ['avec ascenseur', 'avec asc'],
        'Sans_Ascenseur': ['sans asc', 'sans ascenseur'],
        'Duplex': ['duplex', 'sur 2 etages'],
        'Dernier_etage': ['dernier etage'],
        'Premier_etage': ['1er etage', 'premier etage'],
        'Deuxieme_etage': ['2eme etage', 'deuxieme etage'],
        'Troisieme_etage': ['3eme etage', 'troisieme etage'],
        'Quatrieme_etage': ['4eme etage', 'quatrieme etage'],
        'Cinquieme_etage': ['5eme etage', 'cinquieme etage'],
        'Sixieme_etage': ['6eme etage', 'sixieme etage'],
        'Septieme_etage': ['7eme etage', 'septieme etage'],
        'Huitieme_etage': ['8eme etage', 'huitieme etage'],
        'RDC': ['rdc', 'rez de chaussee', 'rez-de-chaussee'],
        'Design': ['architecte', 'design'],
        'Terrasse': ['terrasse', 'balcon', 'loggia'],
        'Parfait_etat': ['refait', 'renove', 'moderne', 'refait a neuf', 'entierement refait', 'parfait etat', 
                        'sans travaux', 'aucun travaux'],
        'Travaux': ['travaux a prévoir', 'rafraichissement à prevoir', 'a rafraichir', 'renovation à prevoir'],
        'Proche_commodites': ['proche commodites', 'proche commerces'],
        'Vue_degagee': ['vue degagee', 'vue entierement degagee', 'vue totalement degagee', 'sans vis-a-vis', 'vis a vis'],
        'Securise': ['securise', 'gardien', 'gardienne', 'gardiens', 'digicode', 'interphone', 'serrure blindee'],
        'Emplacement': ['emplacement', 'idealement situe', 'idealement situee', 'en plein coeur', 'vue sur'],
        'Atypique': ['atelier', 'loft', 'souplex', 'peniche', 'peniche'],
        'Nature': ['arbore', 'arboree', 'nature', 'foret', 'lac', 'montagne', 'verdure', 'verdoyant', 'vignes'],
        'Meuble': ['meuble', 'meublee', 'equipe', 'cuisine amenagee', 'cuisine equipee'],
        'Cave': ['cave'],
        'Parking': ['parking', 'box', 'stationnement', 'place de parking'],
        'Cuisine_americaine': ['cuisine americaine', 'cuisine ouverte', 'cuisine semi ouverte', 'cuisine semi-ouverte'],
        'Investissement_locatif': ['investisseur', 'investissement locatif', 'vendu loue'],
        'Proche_ecoles': ['creche', 'creches', 'ecole', 'ecoles', 'college', 'colleges', 'lycee', 'lycees', 'maternelle'],
        'Dressing': ['dressing'],
        'Plain_Pied': ['plain pied', 'plain-pied', 'plein pied', 'plein-pied'],
        'Garage': ['garage'],
        'Grange': ['grange'],
        'Buanderie': ['buanderie'],
        'Dependance': ['dependance', 'dependances']
    }

    for key, value in immo_features.items():
        add_immo_feature(key, value, nlp, matcher, feature_name_list, color_list, color_pal)
        df[key] = 0

    for feature, color in zip(feature_name_list, color_list):
        colors[feature] = color

    
    # Vérification des entités présentes par défaut
    ner_lst = nlp.pipe_labels['ner']

    # Ajoute les features trouvées pour chaque annonce dans le dataframe
    df['features'] = np.empty((len(df), 0)).tolist()

    ### ATTENTION, ce bloc prend bcp de temps à s'exécuter
    for i in range(0, n_lines):
        doc = nlp(df['desc_clean'][i])
        matches = matcher(doc)

        for match_id, start, end in matches:
            try: 
                span = Span(doc, start, end, label = match_id)
                doc.ents = list(doc.ents) + [span]  # add span to doc.ents
            except:
                pass
        
        ents_features(i, doc, ner_lst, df)

    return df

def prepare_df_model_b(df):
    df = df.drop(columns = {'index', 'link', 'ref', 'agence', 'description', 'details', 'titre', 'taxe_fonciere',
                          'charges', 'orientation', 'classe_energie', 'build_date', 'DPE', 'location', 'desc_clean',
                          'agence_only', 'features'})
    
    # Suppression des annonces avec prix absent
    df = df[df['prix'] != "Nouscontacter"]

    # Les annonces avec Surface Terrain = NA sont forcées à 0m² de terrain
    df['surface_terrain'] = df['surface_terrain'].fillna(0)

    # On supprime les annonces avec nb_chambres = NA
    df = df.dropna(subset = ['nb_chambres'])

    # On encode le type de bien en 1 = Maison, 0 = Appartement
    df['type'] = df['type'].apply(lambda x: 1 if x == "maison" else 0)
    df.isna().sum()
    df = df.dropna()

    # On ne conserve que les biens qui ont moins de 20 (!) pièces ou chambres
    df = df[df['nb_pieces'].isin([str(i) for i in range(20)])]
    df = df[df['nb_chambres'].isin([str(j) for j in range(20)])]

    # Encodage en INT des valeurs numériques qui étaient en str
    df['prix'] = df['prix'].apply(lambda x: x.split(",")[0]).astype(int)
    df['surface'] = df['surface'].apply(lambda x: x.split(",")[0]).astype(int)
    df['surface_terrain'] = df['surface_terrain'].astype(int)
    df['nb_pieces'] = df['nb_pieces'].astype(int)
    df['nb_chambres'] = df['nb_chambres'].astype(int)

    df['prix_m2'] = df['prix'] / df['surface']

    # Suppression des valeurs extrêmes
    prix_max = df['prix'].quantile(0.95)
    prix_min = df['prix'].quantile(0.05)
    surface_max = df['surface'].quantile(0.95)
    surface_min = df['surface'].quantile(0.05)
    prix_m2_max = df['prix_m2'].quantile(0.95)
    prix_m2_min = df['prix_m2'].quantile(0.05)

    logging.info(f"Suppression des valeurs extrêmes | Nb annonces avant filtrage: {df.shape}")
    df = df[(df['prix'] >= prix_min) & (df['prix'] <= prix_max)]
    df = df[(df['surface'] >= surface_min) & (df['surface'] <= surface_max)]
    df = df[(df['prix_m2'] >= prix_m2_min) & (df['prix_m2'] <= prix_m2_max)]
    logging.info(f"Suppression des valeurs extrêmes | Nb annonces après filtrage: {df.shape}")


    # Définition d'un prix median au m² comme base de référence pour voir si une feature a un impact positif ou négatif sur ce prix
    df['dep'] = df['code_postal'].apply(lambda x: x[0:2])

    prix_m2_dep = df.groupby(['dep', 'type'])['prix_m2'].median().reset_index().rename(columns = {'prix_m2' : 'prix_m2_dep'})
    prix_m2_cp = df.groupby(['code_postal', 'type'])['prix_m2'].median().reset_index().rename(columns = {'prix_m2' : 'prix_m2_cp'})
    nb_biens_cp = df.groupby(['code_postal', 'type'])['prix_m2'].count().reset_index().rename(columns = {'prix_m2' : 'nb_biens_cp'})

    df = df.merge(prix_m2_dep, how = "left", on = ["dep", "type"])
    df = df.merge(prix_m2_cp, how = "left", on = ["code_postal", "type"])
    df = df.merge(nb_biens_cp, how = "left", on = ["code_postal", "type"])

    # Le prix m² de reférence est celui du code postal si on a mini 5 biens de référence, ou celui du département sinon
    df['prix_m2_ref'] = df.apply(lambda x: x['prix_m2_cp'] if x['nb_biens_cp'] >= 5 else x['prix_m2_dep'], axis = 1)

    # Calcul du % de différence entre prix/m² du bien et le prix/m² de référence : servira de target car c'est ce qu'on cherche à prédire
    df['evol_prix_m2'] = df.apply(lambda x: round(x['prix_m2'] / x['prix_m2_ref'], 2) - 1, axis = 1)

    # Suppression des évolutions de prix trop extrêmes
    df = df[df['evol_prix_m2'].between(-0.2, 0.2)]

    # Suppression des colonnes inutiles pour génération du modèle
    df = df.drop(columns = {'dep', 'code_postal', 'prix_m2_dep', 'prix_m2_ref', 'nb_biens_cp', 'prix_m2_cp'})

    col = df.columns.to_list()
    col.remove('prix')
    col.remove('surface')
    col.remove('surface_terrain')
    col.remove('nb_pieces')
    col.remove('nb_chambres')
    col.remove('prix_m2')
    col.remove('evol_prix_m2')

    # Calcul de l'influence des features sur le prix au m² et export dans un fichier CSV
    recap_annonces = pd.DataFrame()

    for c in col[1:]:
        for i in ['Tous Biens', 'Appartements', 'Maisons']:
            
            if i == 'Tous Biens':
                prix_moy = df.groupby(c)['prix_m2'].mean().reset_index()
                nb_biens = df.groupby(c)['prix_m2'].count().reset_index().rename(columns = {'prix_m2': 'nb_biens'})
            elif i == 'Appartements':
                prix_moy = df[df['type'] == 0].groupby(c)['prix_m2'].mean().reset_index()
                nb_biens = df[df['type'] == 0].groupby(c)['prix_m2'].count().reset_index().rename(columns = {'prix_m2': 'nb_biens'})
            elif i == 'Maisons': 
                prix_moy = df[df['type'] == 1].groupby(c)['prix_m2'].mean().reset_index()
                nb_biens = df[df['type'] == 1].groupby(c)['prix_m2'].count().reset_index().rename(columns = {'prix_m2': 'nb_biens'})
            
            infos = prix_moy.merge(nb_biens, on = c, how = "left")
            infos['type_local'] = i
            infos['feature'] = c
            infos = infos.rename(columns = {c: "feature_binary"})
            infos['feature_binary'] = infos['feature_binary'].replace([0, 1], ["non", "oui"])

            infos = infos[['feature', 'type_local', 'feature_binary', 'nb_biens', 'prix_m2']]
            infos['prix_m2'] = infos['prix_m2'].astype(int)
            
            recap_annonces = pd.concat([recap_annonces, infos], axis = 0)

    recap_annonces.to_csv(f'models/recap_annonces.csv')

    # Séparation des bases Maisons et Appartements
    df_appart = df[df['type'] == 0]
    df_maison = df[df['type'] == 1]

    return df_appart, df_maison