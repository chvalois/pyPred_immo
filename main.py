from prepare_data import prepare_data
from generate_models_A import generate_models_A
from generate_models_B import generate_models_B

# Génération des modèles A (estimation prix de base) sur la base d'un sample pour tester le code
#prepare_data("2023sample", "2023sample")
#generate_models_A("2023sample")


# Génération des modèles A (estimation prix de base) sur la base des Demandes Valeurs Foncieres 2022 + S1 2023
#prepare_data("all", "20222023s1")
#generate_models_A("20222023s1")


# Génération des modèles B (Bonus/Malus) sur la base de 8000 annonces sélectionnées au hasard
generate_models_B(limit=8000)