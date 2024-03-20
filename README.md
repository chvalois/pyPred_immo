# pyPred_immo

Le projet a pour objectif de déterminer si une annonce immobilière est une bonne affaire ou non en fonction des caractéristiques du bien.

- Pouvoir expliquer ce qui intervient dans la valeur du bien immobilier
- Savoir si l’annonce est sur ou sous estimée par rapport à la réalité du marché
- Pouvoir disposer d’un outil d’estimation automatique

Il est possible d'effectuer des prédictions à partir de l'application Streamlit intégrée dans ce projet :

![2024 03 20 - 6933 - 1920x911](https://github.com/chvalois/pyPred_immo/assets/32735527/a7af5db6-035e-49c0-810f-90ccd8bebdb1)
![2024 03 20 - 6934 - 1920x911](https://github.com/chvalois/pyPred_immo/assets/32735527/0413366f-0518-468e-976a-d2609173a39e)


# Ce qui est fonctionnel

### Lancement des scripts en ligne de commande

Se placer à la racine du projet

Lancer les commandes suivantes : 

Pour la préparation de données (le fichier valeursfoncieres-2023.txt doit se trouver dans databases/dvf) :
```
python -m src --function prepare_data --year 2023 --suffix 2023
```

Pour la génération des modèles A (estimation du prix de base) : 
```
python -m src --function generate_models_A --suffix 2023
```

Pour la génération des modèles B (bonus/malus appliqué en fonction des propriétés du bien) :
```
python -m src --function generate_models_B --limit 5000
```

### Préparation des données

prepare_data.py : 
- Intégration de la base DVF (Demande de Valeurs Foncieres) qui contient des informations sur l'intégralité des biens vendus en France
- Retraitement des adresses pour leur ajouter un code IRIS (code "quartier" INSEE, cf paragraphe plus bas)
- Enrichissement avec base de données BPE Insee (Base Permanente des Equipements) + Indices Loyers + Indices Salaires + Population

### Génération des modèles Machine Learning (Random Forest)
generate_model.py : 
- Génération de 4 modèles pour estimation du prix de base (Modèle A) : 1 global, 1 Maison Province, 1 Appart Province, 1 Appart Paris
- Génération de 2 modèles Bonus/Malus (Maisons, Appartements) qui viennent apporter un bonus/malus au prix de base en fonction des propriétés du bien (Modèle B)

Ces modèles ont fait l'objet de validations croisées (GridSearchCV) et ce sont des modèles de type Random Forest qui ont obtenu les meilleurs résultats.

### Démo Streamlit

demo-streamlit/streamlit-app/streamlit run app.py : Test des modèles via application Streamlit

### Tracking ML Flow

![2024 03 18 - 6929 - 1910x882](https://github.com/chvalois/pyPred_immo/assets/32735527/7c3a0d20-b8f8-403c-b624-fb97c4c99b1b)

### Enrichissement de la base DVF (Demande Valeurs Foncieres) avec les codes IRIS

A partir des adresses des biens immobiliers de la base DVF, il est possible de calculer des coordonnées GPS, et par extension des codes IRIS, et le nombre "d'équipements à la maille quartier IRIS" (ex. : nb de restaurants)
Ces statistiques de nombre d'équipements sont utilisées dans la génération du modèle A d'estimation de prix.

Le code IRIS est un code numérique de 9 chiffres dont les cinq premiers correspondent au code Insee de la commune.
Afin de préparer la diffusion du recensement de la population de 1999, l'INSEE avait développé un découpage du territoire en mailles de taille homogène appelées IRIS2000. Un sigle qui signifiait « Ilots Regroupés pour l'Information Statistique » et qui faisait référence à la taille visée de 2 000 habitants par maille élémentaire. Depuis, l'IRIS (appellation qui se substitue désormais à IRIS2000) constitue la brique de base en matière de diffusion de données infra-communales. Il doit respecter des critères géographiques et démographiques et avoir des contours identifiables sans ambigüité et stables dans le temps.

Les communes d'au moins 10 000 habitants et une forte proportion des communes de 5 000 à 10 000 habitants sont découpées en IRIS.

### Identification des propriétés du bien décrit dans l'annonce pour mise en place du modèle B (Bonus/Malus)

Sur des milliers d'annonces scrapées, des annotations ont été réalisées pour déterminer le rapport entre propriétés du bien (exemple : l'annonce fait état d'une exposition Sud) et le prix demandé du bien.
Cela a permis d'entraîner un modèle pour la vente de maison, et un modèle pour la vente d'appartement, qui va venir augmenter ou diminuer le prix de base estimé par le modèle A 

![2024 03 19 - 6931 - 1434x710](https://github.com/chvalois/pyPred_immo/assets/32735527/726d6e58-2373-49bc-8ccc-fa8a4b827307)

# Bases de données utilisées

Plus d'informations dans references/references.txt

Fichiers DVF (Demandes Valeurs Foncieres)

https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/


Base Permanente des Equipements

Page : https://www.insee.fr/fr/statistiques/3568629?sommaire=3568656

Fichier : https://www.insee.fr/fr/statistiques/fichier/3568629/Contenu_bpe21_ensemble.zip


Contours IRIS

https://geoservices.ign.fr/contoursiris#telechargement


Diagnostics de Performance Energétique

https://data.ademe.fr/datasets/dpe-france


Indicateurs de Loyers 2022

https://www.data.gouv.fr/fr/datasets/carte-des-loyers-indicateurs-de-loyers-dannonce-par-commune-en-2022/


Base Recensement de la Population 2020

https://www.insee.fr/fr/statistiques/7632446?sommaire=7632456


Base Revenus par code IRIS 2020

https://www.insee.fr/fr/statistiques/7233950


Base Revenus des ménages 2018

https://www.data.gouv.fr/fr/datasets/revenus-et-pauvrete-des-menages-aux-niveaux-national-et-local-revenus-localises-sociaux-et-fiscaux/


Base Annonces Scrapées (Non communiqué)



# Reste à faire

- Réflexion mise à disposition des databases de départ (non disponibles sur Github et nécessaires à la préparation de data, et génération de modèle)
- Créer docker-compose + dockerfile
- Tester PCA sur données BPE (base permanente des équipements) pour réduction de dimensions
- Mettre à disposition une API
- Ajouter des tests unitaires
- Ajouter parallélisation


