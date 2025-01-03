import streamlit as st
from PIL import Image
import os

st.set_page_config(
        page_title='Conclusion et Perspective',
        layout="wide"
    )


image_path = os.path.join('Photo', 'Bannière accident.png')
banniere = Image.open(image_path)
st.image(banniere, use_column_width="always")

st.markdown("""
            <h1>
            12. Prédire la gravité des accidents de la route ?
            </h1>
            """, 
            unsafe_allow_html=True)

st.markdown("""
            La gravité des accidents de la route en France est un sujet de préoccupation pour les autorités,
            les citoyens et les associations de prévention routière. Les accidents de la route peuvent entraîner
            des conséquences graves, notamment des décès, des blessures corporelles et des dégâts matériels.
            
            D'après l'Observatoire national interministériel de la sécurité routière, plusieurs facteurs contribuent
            à la gravité des accidents de la route en France, notamment la vitesse excessive, l'alcool, la drogue, la fatigue,
            le non-respect des règles de circulation et le manque d'entretien des véhicules. Les comportements irresponsables
            des usagers de la route augmentent le risque d'accidents graves. Et c'est précisément pourquoi,
            puisque nous ne disposons pas de ces facteurs explicatifs dans notre jeu de données, que notre meilleur
            modèle de prédiction n'a pas, malgré tout, d'excellents résultats puisque l'Accuracy n'est que de 66%.
            """)
st.write("")

st.markdown("""
            <h1>
            13. Et pour aller plus loin ?
            </h1>
            """, 
            unsafe_allow_html=True)

st.markdown("""
            Il reste encore des points à approfondir ou des nouvelles méthodes à appliquer pour perfectionner ce projet :
            
            - Développer un scoring des zones à risque en fonction des informations météorologiques des emplacements géographiques ;
            - Faire attention au biais dans la donnée. Le dataset est issu des formulaires remplis par les
              forces de l’ordre lors d’un accident. Il peut y avoir un facteur humain introduisant un biais dans la donnée ;
            - Faire un encodage cyclique sur les variables temps (créneau horaire par exemple) ;
            - Mettre en lien la proportion des accidents par rapport au pourcentage de la population par région.
              Cela peut permettre de faire apparaitre d’autres corrélations ou nuancer certains résultats ;
            - Utiliser des variables non catégorielles (département, âge, vitesse, …) ;
            - Utiliser l’ACP pour réduire le nombre de variables ;
            - Compléter le jeu de données. Le dataset ne contient que des informations sur les accidents.
              Il serait intéressant de voir la proportion de gens véhiculés par exemple, les taux d’alcoolémie, …
            - Approfondir le traitement des données déséquilibrées : meilleure compréhension de l’algorithme SMOTE et ses variantes (SMOTE-NC) ;
            - Approfondir l’évaluation des modèles : classification multiclasses, utilisation des métriques multiclasses plus adaptées ;
            - Approfondir l’analyse des graphiques d’interprétation des modèles multiclasses (Force plot, Decision plot).
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

