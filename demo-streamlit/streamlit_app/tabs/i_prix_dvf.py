import streamlit as st
import pandas as pd
import numpy as np
import os


title = "Visualisation prix DVF"
sidebar_name = "1. Visualisation prix DVF"


def run():


    st.title('Visualisation des prix au m2 par Commune (2020)')
    
    st.write('Les prix indiqués dans la cartographie ci-dessous ont été calculés sur la base des fichiers de Demandes Valeurs Foncières')
    st.write('Ces fichiers permettent de connaître les transactions immobilières intervenues au cours des cinq dernières années sur le territoire métropolitain et les DOM-TOM, à l’exception de l’Alsace, de la Moselle et de Mayotte.')    
    
    st.write('Sélectionnez un département et le type de bien qui vous intéresse pour connaître les prix moyens au m² par commune en 2022 :')

    df = pd.read_parquet('../../databases/temp/20222023s1_05_export_dvf_completed_final.parquet')
          
    # ------ Calcul du prix moyen au m2 ------
    
    prix_moyen_commune = df.groupby(['Code commune INSEE', 'Code postal', 'Commune'])['prix_m2'].mean().reset_index()
    
    prix_maison = df[df['Type local'] == "Maison"].groupby(['Code commune INSEE', 'Code postal', 'Commune'])['prix_m2'].mean().reset_index()
    prix_maison = prix_maison.rename(columns = {'prix_m2': 'prix_m2_maison'})
    
    prix_appart = df[df['Type local'] == "Appartement"].groupby(['Code commune INSEE', 'Code postal', 'Commune'])['prix_m2'].mean().reset_index()
    prix_appart = prix_appart.rename(columns = {'prix_m2': 'prix_m2_appart'})
    
    prix_moyen_commune = prix_moyen_commune.merge(prix_maison, on = ['Code commune INSEE', 'Code postal', 'Commune'], how = "left")
    prix_moyen_commune = prix_moyen_commune.merge(prix_appart, on = ['Code commune INSEE', 'Code postal', 'Commune'], how = "left")
    
    # Formulaire
    form = st.form("viz_prix")

    dep = form.text_input('Département', value = '33')
    dep = str(dep)

    type_bien = form.radio('Type de bien', options = ['Maisons et Appartements', 'Maisons', 'Appartements'])
    
    form.form_submit_button("Visualiser les prix")
    
    #st.write('Communes du département ' + dep + ' avec les prix au m2 les plus élevés :')
    #st.write(prix_moyen_commune[prix_moyen_commune['DEPCOM'].str.startswith(dep)].sort_values(by = 'prix_m2', ascending = False))
    
    
    
    
    # ------ Création de la cartographie ------
    
    # Import des coordonnées GPS par département pour centrer la cartographie
    gps_dep = pd.read_csv('../../databases/GPS_departement.csv', sep = ";")
    dep_latitude = gps_dep[gps_dep['Code departement'] == dep]['Latitude']
    dep_longitude = gps_dep[gps_dep['Code departement'] == dep]['Longitude']
    dep_nom = gps_dep[gps_dep['Code departement'] == dep]['Nom departement']
    
    st.write('Prix au m² des communes du département ' + dep)
    
    import folium
    from folium.features import GeoJson, GeoJsonTooltip, GeoJsonPopup
    from streamlit_folium import folium_static
    import geojson, geopandas
    
    # Lecture des données géographiques par commune
    c = geopandas.read_file("../../databases/shapefiles/communes-20210101.shp")
    c = c.drop(columns = {'wikipedia'})
    
    # Ajout des arrondissements de Paris, Lyon et Marseille
    a_all = geopandas.read_file("../../databases/shapefiles/arrondissements-millesimes0.shp", encoding='utf-8')
    a_all = a_all[['code_insee', 'superficie', 'nom_com', 'geometry']]
    a_all = a_all.rename(columns = {'code_insee' : 'insee', 'superficie': 'surf_ha', 'nom_com': 'nom'})
    a_all = a_all.reindex(columns=['insee','nom','surf_ha','geometry'])
    
    # Concaténation des deux dataframes géographiques
    c = pd.concat([a_all, c], axis = 0, ignore_index=True)
    c = c.rename(columns = {'insee' : 'DEPCOM'})
    
    # Suppression des codes communes correspondant à Paris, Marseille et Lyon
    c = c[c['DEPCOM'] != '13055']
    c = c[c['DEPCOM'] != '69123']
    c = c[c['DEPCOM'] != '75056']
    
    # Ajout du prix_m2 au dataframe
    prix_moyen_commune = prix_moyen_commune.rename(columns = {'Code commune INSEE': 'DEPCOM'})
    c = c.merge(prix_moyen_commune, on = "DEPCOM", how = "left")
    
    # Arrondi du prix
    c['prix_m2'] = c['prix_m2'].round(0)
    
    # Filtre sur département choisi par l'utilisateur
    c_selec = c[c['DEPCOM'].str.startswith(dep)]
    
    st.write(c_selec.head())
    st.write(prix_moyen_commune.head())

    with open("c_selec.geojson", "w") as f:
        geojson.dump(c_selec, f)
      
        
    # Map
    
    if type_bien == "Maisons":
        fields = ['Commune', 'prix_m2_maison']
        columns = ['DEPCOM', 'prix_m2_maison']
    elif type_bien == "Appartements":
        fields = ['Commune', 'prix_m2_appart']
        columns = ['DEPCOM', 'prix_m2_appart']
    else:
        fields = ['Commune', 'prix_m2']
        columns = ['DEPCOM', 'prix_m2']
    
    
    coords = (dep_latitude,dep_longitude)
    
    f = folium.Figure(width=680, height=750)
    
    map = folium.Map(location=coords, tiles='OpenStreetMap', zoom_start=9).add_to(f)
    
    popup = GeoJsonPopup(
        fields=['Commune','prix_m2'],
        aliases=['Commune',"Prix au m² : "], 
        localize=True,
        labels=True,
        style="background-color: yellow;",
    )
    
    tooltip = GeoJsonTooltip(
        fields=['Commune','prix_m2'],
        aliases=['Commune',"Prix au m² : "],
        localize=True,
        sticky=False,
        labels=True,
        style="""
            background-color: #F0EFEF;
            border: 1px solid black;
            border-radius: 3px;
            box-shadow: 3px;
        """,
        max_width=800,
    )
    
    g = folium.Choropleth(
        geo_data = 'c_selec.geojson',
        name = 'choropleth',
        data = c_selec,
        columns = ['DEPCOM', 'prix_m2'], # data key/value pair
        key_on = 'feature.properties.DEPCOM', # corresponding layer in GeoJSON
        #threshold_scale = [0, 2000, 4000, 6000, 8000, 10000, 20000],
        fill_color = 'OrRd',
        fill_opacity = 0.8,
        line_opacity = 0.2,
        legend_name = 'Prix au m2'
    ).add_to(map)
    
    folium.GeoJson(
        'c_selec.geojson',
        style_function=lambda feature: {
            'fillColor': '#ffff00',
            'color': 'black',
            'weight': 0.2,
            'dashArray': '5, 5'
        },
        tooltip=tooltip,
        popup=popup).add_to(g)
    
    
    folium_static(map)

