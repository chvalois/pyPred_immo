# projet_immo
Le projet a pour objectif de déterminer si une annonce immobilière est une bonne affaire ou non en fonction des caractéristiques du bien.

- Pouvoir expliquer ce qui intervient dans la valeur du bien immobilier
- Savoir si l’annonce est sur ou sous estimée par rapport à la réalité du marché
- Pouvoir disposer d’un outil d’estimation automatique

![image](https://github.com/chvalois/pyPred_immo/assets/32735527/2bb1466e-85b6-4683-b423-50c02ba87c31)

# Fonctionnel

- prepare_data.py : Préparation des données (intégration de la base DVF, retraitement des adresses pour leur ajouter un code IRIS (code "quartier" INSEE), ajout BDD BPE+Loyers+Salaires+Population)
- generate_model.py : Génération de 4 modèles Random Forest : 1 global, 1 Maison Province, 1 Appart Province, 1 Appart Paris
- demo-streamlit/streamlit-app/streamlit run app.py : Test des modèles via application Streamlit

# Reste à faire

- Réflexion mise à disposition des databases de départ (non disponibles sur Github et nécessaires à la préparation de data, et génération de modèle)
- Ajouter la possibilité de lancer la préparation de données (prepare_data.py) et la génération de modèles (generate_model.py) en ligne de commande
- Créer docker-compose + dockerfile
- Tester PCA sur données BPE (base permanente des équipements) pour réduction de dimensions
- Mettre à jour le modèle B à partir des annonces scrapées (modèles actuellement en production ne sont pas compatibles avec version scikit learn installée), evtl prévoir un autre environnement pour ce modèle
- Mettre à disposition une API
- Ajouter des tests unitaires
- Ajouter parallélisation


