# **dataViz_2020-2022_rework**

Ce fichier est un notebook Jupyter que j’utilise pour analyser et visualiser des données couvrant la période 2020-2022. Il commence par l’importation des bibliothèques nécessaires, comme pandas pour manipuler les données et matplotlib ou seaborn pour créer des graphiques. Ensuite, j’y charge des fichiers de données au format .csv, et je procède à un nettoyage et à une transformation des données, par exemple en traitant les valeurs manquantes ou en ajustant les types de colonnes. Le notebook contient aussi des visualisations pour explorer les tendances et identifier des informations importantes. J’y effectue des analyses exploratoires, avec des statistiques descriptives, et, si nécessaire, je mets en place des modèles prédictifs ou des algorithmes de machine learning. Bref, c’est un outil que j’utilise pour mieux comprendre les relations entre les variables et en tirer des insights pertinents.

# **dataPreprocess_2020-2022**

Ce fichier est un notebook Jupyter que j’utilise pour préparer les données couvrant la période 2020-2022. Dans celui-ci je commence par supprimer les variables avec trop de "NaN" et je corrige les anomalies ou valeurs aberrantes comme par exemple enlever les limitations de vitesse supérieure à 130 km/h. Enfin j'affiche des visuels suivant différentes variables comme le pourcentage d'accidents par jour de semaine ou encore le nombre d'accidents par jour.

# **Classification_2020-2022**

Ce fichier est un notebook Jupyter que j’utilise pour entraîner et évaluer un modèle de classification sous RandomForestClassifier couvrant la période 2020-2022. Dans ce fichier, j'importe mon jeu de données 2020-2022 puis je le découpe en deux jeux de données, un jeu d'entraînement et un jeu de test. Ensuite je défini une grille d'hyper-paramètres à tester puis j'entraine RandomizedSearchCV pour trouver le modèle le plus performant sous RandomSearch. Enfin j'évalue le modèle pour avoir un score "accuracy" et un classification report pour avoir des informations complémentaires sur le modèle le plus performant.
