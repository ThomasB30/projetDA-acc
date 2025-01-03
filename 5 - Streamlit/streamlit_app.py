import streamlit as st
from PIL import Image
import os

# Configuration de la page
st.set_page_config(
        page_title='Les accidents routiers en France',
        layout="wide")


# tous les chargements (images, data et modeles) se font via os
image_path = os.path.join('Photo', 'Bannière accident.png')
banniere = Image.open(image_path)
st.image(banniere, use_column_width="always")

# Titre et sous_titre du projet
st.markdown("""
            <p style="color:Red;text-align:center;font-size:2em;font-style:italic;font-weight:700;margin:0px;">
            Projet Fil Rouge - DataAnalyst - Mai 2024<br>
            """, 
            unsafe_allow_html=True)

st.write("")
st.markdown("""*Repository Github du projet : [cliquez ici](https://github.com/ThomasB30/projetDA-acc)*""")
st.write("")

# Description et Objectif du projet
# Titre 1
st.write("")
st.markdown("""
            <h1>
            1. Contexte et Objectifs
            </h1>
            """, 
            unsafe_allow_html=True)
st.write("")

st.markdown("""
            Chaque année depuis 2005, l'Observatoire National Interministériel de la Sécurité Routière (ONISR)
            met à disposition (via le site data.gouv) des bases de données relatives aux accidents corporels de la circulation routière.
            
            Pour chaque accident corporel (accident survenu sur une voie ouverte à la circulation publique,
            impliquant au moins un véhicule et causant au moins une victime), des informations sont recueillies
            par les forces de l'ordre présentes sur les lieux. Ces informations sont consignées dans un document
            appelé bulletin d’analyse des accidents corporels. L'ensemble de ces bulletins constitue le fichier national
            des accidents corporels de la circulation, communément appelé « Fichier BAAC », géré par l'ONISR.
            
            Les bases de données issues de ce fichier recensent tous les accidents corporels survenus au cours d'une année donnée
            en France métropolitaine, dans les départements et territoires d'Outre-mer avec une description simplifiée.
            Elles contiennent des informations sur la localisation de l'accident, les caractéristiques de l'accident et du lieu,
            ainsi que sur les véhicules impliqués et leurs victimes, qui sont rangées dans quatre bases de données : usagers,
            véhicules, lieux et caractéristiques.
            """)
st.write("")

st.markdown("""
            L’objectif de ce projet est de se familiariser avec un projet de Machine Learning et de mettre en pratique les cours.
            Des modèles seront construits et utilisés pour essayer de prédire la gravité des accidents routiers en France.
            Les prédictions seront basées sur les données de l’ONISR. Le métier de Data Analyst consistant aussi à interpréter les données,
            une partie du projet sera orienter sur la compréhension des résultats des modèles et pouvoir répondre à la question
            ***« pourquoi le modèle a fait cette prédiction ? »***.
            """)

st.write("")
st.markdown("""
            <h1>
            2. Les Données
            </h1>
            """, 
            unsafe_allow_html=True)

st.markdown("""
            Bien que l’on puisse penser qu’il suffit d’un grand nombre de données pour avoir un algorithme performant, 
            les données dont nous disposons sont souvent non adaptées. 
            Il faut donc les comprendre et les traiter préalablement pour pouvoir ensuite les utiliser : 
            c’est l’étape d'exploration et de visualisation des données.
            
            En effet, des erreurs d’acquisition liées à des fautes humaines ou techniques peuvent corrompre 
            notre dataset et biaiser l’entraînement. 
            Parmi ces erreurs, nous pouvons citer des informations incomplètes, des valeurs manquantes ou erronées.
            
            Il est donc souvent indispensable d’établir une stratégie de pré-traitement des données à partir des données brutes 
            pour arriver à des données exploitables qui nous donneront un modèle plus performant.
            
            La particularité de notre jeu de données est que nous avons 4 fichiers par année avec les mêmes variables :
            - Fichier Caractéristiques: Ce fichier décrit les circonstances générales de chaque accident
            - Fichier Lieux : Ce fichier décrit le lieu principal de l'accident
            - Fichier Véhicules : Ce fichier décrit les véhicules impliqués dans l'accident
            - Fichier Usager : Ce fichier décrit les personnes impliquées dans l'accident
            
            Notre objectif étant de prédire la gravité d’un accident, nous avons choisi de limiter
            notre jeu de données à trois années, soit de 2020 à 2022.
            
            Ce choix est motivé par le souhait d’optimiser nos performances matérielles, car nous travaillons
            déjà sur un volume de données conséquent, comptant 361 205 lignes.
            
            Par ailleurs, nous avons estimé que la gravité d’un accident ne pouvait pas être comparée de manière uniforme
            entre aujourd’hui et 2005. Cela s’explique notamment par les évolutions dans les travaux de voirie et les progrès technologiques des véhicules.
""")


# Centrage de l'image du logo dans la sidebar
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.sidebar.write("")
with col2:
    image_path = os.path.join('Photo', 'logo-datascientest.png')
    logo = Image.open(image_path)
    st.sidebar.image(logo, use_column_width="always")
with col3:
    st.sidebar.write("")



