import pandas as pd
import numpy as np
from joblib import load
import streamlit as st

from tabs.functions.features_map import features_map
from tabs.functions.iris_functions import get_iris_oneaddress

import geopy.geocoders
from geopy.geocoders import Nominatim
import certifi
import ssl

ctx = ssl.create_default_context(cafile=certifi.where())
geopy.geocoders.options.default_ssl_context = ctx

title = "demo"
sidebar_name = "4. Démo du modèle"

# initialise a boolean attr in session state
if "button" not in st.session_state:
    st.session_state.button = True

# write a function for toggle functionality
def toggle():
    if st.session_state.button:
        st.session_state.button = False
    else:
        st.session_state.button = True

def run():

    suffix = "20222023s1"
    st.title('pyPredImmo - Votre futur bien immobilier est-il une bonne affaire ?')
    
    # Applique-t-on le bonus/malus ?
    application_bonus = 1   
    
    # Renseigner ici les propriétés du bien    
    with st.expander("Propriétés du bien à estimer", expanded=st.session_state.button):

        col1, col2 = st.columns(2)

        with col1:
            prix_annonce = st.number_input("Prix du bien (hors frais d'agence) en euros : ", value = 399000, min_value = 0, max_value = 2000000, step = 5000)    
            rue = st.text_input("Nom de la rue :", value = 'rue Montaigne')
            cp = st.number_input("Code postal à 5 chiffres:", value = 33170, min_value = 1000, max_value = 99000)
            ville = st.text_input("Ville:", value = "Gradignan")
            surface = st.number_input("Surface du bien (m²): ", value = 114, min_value = 0, max_value = 5000, step = 5)
            type_local_str = st.radio('Type de bien :', options = ['Appartement', 'Maison'], index = 1)

        with col2:

            nb_pieces = st.number_input("Nombre de pièces : ", value = 5, min_value = 1, max_value = 20, step = 1)
        
            date_construction = st.number_input("Quand a été construit le bien ?", value = 1987, min_value = 1800, max_value = 2022)
            conso_energie = st.number_input("Quelle est la consommation énergétique du bien ?", value = 187, min_value = 0, max_value = 2000, help = 'de < 50 (classe A économe) à > 450 (classe G énergivore)')
            estim_ges = st.number_input("Quelle est la note GES du bien (kg eq. CO2 / m² / an) ?", value = 6, min_value = 0, max_value = 500, help = 'de <=5 (faible émission GES) à > 80 (forte émission GES)')
            surface_terrain = st.number_input("Surface du terrain (m²): ", value = 846, min_value = 0, max_value = 100000, step = 100, help = "Indiquer 0 si appartement ou pas de terrain")
            mer_str = st.radio('Le bien est-il près de la mer (< 20 kms) :', options = ['Oui', 'Non'], index = 1)


        extra_features = st.multiselect("Quelles sont les spécificités du bien ?", 
                                        help = "Vous pouvez sélectionner aucune, une ou plusieurs spécificités",
                                        options = ['Exposition Sud', 'Lumineux', 'Calme', 'Beaux volumes', 
                                                'Standing', 'Charme', 'Mezzanine', 'Avec Ascenseur', 
                                                'Sans Ascenseur', 'Duplex', 'Dernier étage', 'Premier étage', 'Deuxième étage', 'Troisième étage', 
                                                'Quatrième étage', 'Cinquième étage', 'Sixième étage', 'Septième étage', 'Huitième étage', 'RDC', 'Design', 
                                                'Terrasse', 'Parfait état', 'Travaux à prévoir', 'Proche commodités', 'Vue dégagée', 'Sécurisé', 'Emplacement', 
                                                'Atypique', 'Proche de la nature', 'Meublé', 'Cave', 'Parking', 'Cuisine américaine', 'Investissement locatif', 
                                                'Proche écoles', 'Dressing', 'Plain Pied', 'Garage', 'Grange', 'Buanderie', 'Dépendance'],   
                                        default = ['Exposition Sud', 'Lumineux', 'Garage', 'Mezzanine', 'Terrasse', 'Parking', 'Proche écoles', 'Proche commodités'])        
                                        
        #form.form_submit_button("Valider")
        submitted = st.button("Submit", on_click=toggle)

    if submitted:

        with st.spinner():

            my_bar = st.progress(0, text="Chargement des modèles")

            # Retraitement des features de l'annonce pour qu'elles soient compatibles avec le modèle de prédiction
            if date_construction > 2024:
                anciennete = 0
            else: 
                anciennete = 2024 - date_construction
            
            if (str(cp)[0:2] == "75") & (len(str(cp)) == 5):
                paris = 1
            else:
                paris = 0
            
            if mer_str == "Oui":
                mer = 1
            else:
                mer = 0
            
            if type_local_str == "Appartement":
                type_local = 0
            else:
                type_local = 1
            
            if paris == 1:
                model_A = load(f'assets/models/{suffix}_model_A_2_appart_paris.pkl')
            elif type_local == 0:
                model_A = load(f'assets/models/{suffix}_model_A_1_appart_province.pkl')
            elif type_local == 1:
                model_A = load(f'assets/models/{suffix}_model_A_3_maison_province.pkl')
            
            # Chargement des modèles A
            #model_A_all = load(f'assets/models/{suffix}_model_A_all.pkl')
            #model_A_appart_province = load(f'assets/models/{suffix}_model_A_1_appart_province.pkl')
            #model_A_appart_paris = load(f'assets/models/{suffix}_model_A_2_appart_paris.pkl')
            #model_A_maison_province = load(f'assets/models/{suffix}_model_A_3_maison_province.pkl')    
            
            # Chargement de la liste des features requis pour faire tourner le modèle
            # df_feat = pd.read_csv('assets/models/features_list_for_model_A.csv', index_col = 0)

            df_feat = ['Paris', 'Type local', 'Surface reelle bati', 'Nombre pieces principales', 
            'Surface terrain', 'littoral', 'loyer', 'nb_habitants', 'revenu', 'college_lycee', 'creche', 'ecole', 'ecole_sup',
            'gare', 'hotels', 'info_touristique', 'police', 'restaurants', 'salle_multisport', 'anciennete_vente', 'prix_m2_commune', 'prix_m2_gps']
            

            # Complétion des informations          
            my_bar.progress(0.2, text="Recherche des coordonnées GPS")
            adresse = rue + " " + str(cp) + " " + ville
            #code_dep = str(cp)[0:2]
            
            geolocator = Nominatim(user_agent="pyPred_immo")
            
            try:
                location = geolocator.geocode(adresse, country_codes = 'fr', timeout = 1)

                mapping_gps_iris = pd.read_parquet('assets/mapping_gps_iris.parquet')
                lon = round(location.longitude, 2)
                lat = round(location.latitude, 2)
                
                try:
                    iris_annonce = mapping_gps_iris[(mapping_gps_iris['lon_2'] == lon) & (mapping_gps_iris['lat_2'] == lat)]['code_iris'].values[0]
                except:
                    iris_annonce = 'non trouvé'
                st.write(f"Coordonnées GPS de l'adresse '{adresse}' : {str(lat)}, {str(lon)}, code IRIS : {iris_annonce}")
            
            except Exception as err:
                st.write(err)
                st.write('L\'adresse du bien n\'a pas été trouvée')
            
            prix_m2_commune = pd.read_csv(f'assets/models/{suffix}_prix_m2_commune.csv', dtype = {'Code postal': int})
            prix_m2_gps = pd.read_csv(f'assets/models/{suffix}_prix_m2_gps.csv')
            
            prix_m2_commune_bien = prix_m2_commune[(prix_m2_commune['Code postal'] == cp) & (prix_m2_commune['Type local'] == type_local_str)]
            prix_m2_commune_bien_val = prix_m2_commune_bien['Prix m2'].iloc[0]
            
            try:
                prix_m2_gps_bien = prix_m2_gps[(prix_m2_gps['lon_2'] == lon) & (prix_m2_gps['lat_2'] == lat) & (prix_m2_gps['Type local'] == type_local_str)]
                prix_m2_gps_bien_val = prix_m2_gps_bien['Prix m2'].iloc[0]
            except:
                prix_m2_gps_bien_val = prix_m2_commune_bien_val
            
            # Recherche du loyer, population et revenus
            df_other_feat = pd.read_csv('assets/models/loy_rev_pop.csv', dtype = {'Code postal': int}, index_col = 0)
            df_other_feat_bien = df_other_feat[(df_other_feat['Code postal'] == cp) & (df_other_feat['Type local'] == type_local_str)]

            loyer = df_other_feat_bien['loyer'].iloc[0]
            pop = df_other_feat_bien['Population totale'].iloc[0]
            revenu = df_other_feat_bien['SNHMO18'].iloc[0]
            
            # Récupération du code IRIS et des équipements du quartier IRIS
            my_bar.progress(0.5, text="Recherche du code IRIS")

            if iris_annonce == 'non trouvé':
                try:
                    iris_annonce = get_iris_oneaddress(lon, lat)
                except Exception as err:
                    st.write(err)
                    iris_annonce = 0
                
            bpe = pd.read_parquet(f'assets/models/{suffix}_04b_bpe.parquet')
            
            if iris_annonce == 0:
                college_lycee = 0
                creche = 0
                ecole = 0
                ecole_sup = 0
                gare = 0
                hotels = 0
                info_tour = 0
                police = 0
                restaurants = 0
                salle_sport = 0
            else:    
                college_lycee = int(bpe[bpe['code_iris'] == iris_annonce]['college_lycee'])
                creche = int(bpe[bpe['code_iris'] == iris_annonce]['creche'])
                ecole = int(bpe[bpe['code_iris'] == iris_annonce]['ecole'])
                ecole_sup = int(bpe[bpe['code_iris'] == iris_annonce]['ecole_sup'])
                gare = int(bpe[bpe['code_iris'] == iris_annonce]['gare'])
                hotels = int(bpe[bpe['code_iris'] == iris_annonce]['hotels'])
                info_tour = int(bpe[bpe['code_iris'] == iris_annonce]['info_touristique'])
                police = int(bpe[bpe['code_iris'] == iris_annonce]['police'])
                salle_sport = int(bpe[bpe['code_iris'] == iris_annonce]['salle_multisport'])
                restaurants = int(bpe[bpe['code_iris'] == iris_annonce]['restaurants'])
            
            
            # Récap des propriétés du bien pour prédiction du modèle A et calcul des prédictions
            my_bar.progress(0.7, text="Calcul de l'estimation du prix de base")
            
            annonce_proprietes_A = [paris, type_local, surface, nb_pieces, surface_terrain, mer, loyer, pop, revenu, 
                                    college_lycee, creche, ecole, ecole_sup, gare, hotels, info_tour, police, restaurants, salle_sport,
                                    anciennete, prix_m2_commune_bien_val, prix_m2_gps_bien_val]
            

            annonce_proprietes_A = np.array(annonce_proprietes_A).reshape(1, len(df_feat))
                
            pred_model_A = model_A.predict(annonce_proprietes_A)
            prix_base_A = round(pred_model_A[0], 0)

            # Calcul du bonus / malus du modèle B

            # Chargement des modèles B
            my_bar.progress(0.9, text="Calcul du bonus/malus en fonction des propriétés du bien à estimer")

            if type_local == 0:
                model_B = load(f'assets/models/model_B_features_appart.pkl')
            else:
                model_B = load(f'assets/models/model_B_features_maison.pkl')
            
            # Chargement de la liste des features requis pour faire tourner le modèle
            df_feat_b = pd.read_csv(f'assets/models/features_list_for_model_B.csv', index_col = 0)
            
            # Remplissage des paramètres à partir du multi-select extra_features
            Exposition_Sud, Lumineux, Calme, Volume, Standing, Charme, Mezzanine, Avec_Ascenseur, Sans_Ascenseur, Duplex, Dernier_etage, Premier_etage, Deuxieme_etage, Troisieme_etage, Quatrieme_etage, Cinquieme_etage, Sixieme_etage, Septieme_etage, Huitieme_etage, RDC, Design, Terrasse, Parfait_etat, Travaux, Proche_commodites, Vue_degagee, Securise, Emplacement, Atypique, Nature, Meuble, Cave, Parking, Cuisine_americaine, Investissement_locatif, Proche_ecoles, Dressing, Plain_Pied, Garage, Grange, Buanderie, Dependance = features_map(extra_features)   
            
            annonce_proprietes_B = [type_local, Exposition_Sud, Lumineux, Calme, Volume, Standing, Charme, Mezzanine, Avec_Ascenseur, Sans_Ascenseur, Duplex,
            Dernier_etage, Premier_etage, Deuxieme_etage, Troisieme_etage, Quatrieme_etage, Cinquieme_etage, Sixieme_etage, Septieme_etage,
            Huitieme_etage, RDC, Design, Terrasse, Parfait_etat, Travaux, Proche_commodites, Vue_degagee, Securise, Emplacement, Atypique, Nature, Meuble, Cave, Parking,
            Cuisine_americaine, Investissement_locatif, Proche_ecoles, Dressing, Plain_Pied, Garage, Grange, Buanderie, Dependance]
            
            annonce_proprietes_B = np.array(annonce_proprietes_B).reshape(1, df_feat_b.shape[0])

            bonus_malus_B = model_B.predict(annonce_proprietes_B)
            
            if bonus_malus_B > 0:
                bonus_malus_B_str = "+" + "{0:.2%}".format(float(bonus_malus_B))
            else:
                bonus_malus_B_str = "{0:.2%}".format(float(bonus_malus_B))


            if application_bonus == 1:
                prix_predict_final = prix_base_A * (1 + (bonus_malus_B))
            else:
                prix_predict_final = prix_base_A

            # Fonction qui détermine la jauge affichée en fonction de la différence entre prédiction et prix de l'annonce
            
            def analyse_prix(prix_annonce, prix_predict_final):
                
                diff = prix_annonce / prix_predict_final - 1
            
                if diff > 0.1:
                    st.image('assets/jauge_1.png')
                elif diff > 0.05:
                    st.image('assets/jauge_2.png')
                elif diff > -0.05:
                    st.image('assets/jauge_3.png')
                elif diff > -0.1:
                    st.image('assets/jauge_4.png')
                else:
                    st.image('assets/jauge_5.png')

            
            # Affichage des résultats
            my_bar.progress(0.95, text="Affichage des résultats")

            st.subheader('Décomposition de la prédiction de prix')
            st.write("Prédiction du prix de base (modèle basé sur la base DVF): ", prix_base_A, "€")
            st.write("Prédiction du bonus/malus basé sur les spécificités du bien (modèle basé sur les annonces immo): ", bonus_malus_B_str)

            st.subheader('Analyse finale du prix du bien')
            st.write("Prix de l'annonce :", int(prix_annonce), "€", "(prix au m² : ", round(int(prix_annonce) / surface, 0), "€)")
            st.write("Prédiction de prix :", int(prix_predict_final), "€", "(prix au m² : ", round(int(prix_predict_final) / surface, 0), "€)")
            st.write('\n')
            st.write("Résultat :")
            analyse_prix(prix_annonce, prix_predict_final)

            my_bar.empty()

        
            
                
                