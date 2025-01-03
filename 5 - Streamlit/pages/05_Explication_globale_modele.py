import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import pickle
from PIL import Image
import os


# Configuration de la page
st.set_page_config(
        page_title='Interprétation Globale de la prédiction',
        layout="wide" )

# Lecture des fichiers
# Chargement du modèle
model_path = os.path.join("Modeles", "best_model_LGBM_resampled.pkl")
model = pickle.load(open(model_path, "rb"))
target_names = ["indemne", "tué", "blessé_hospitalisé", "blessé_léger"] #list(model.classes_)
# Rappel des numéros des classes et de leur libellé
# 1:"indemne"
# 2:"tué"
# 3:"blessé_hospitalisé"
# 4:"blessé_léger"

# Chargement des valeurs Shap
shap_path = os.path.join('Modeles', 'shap_values_lgbm_all.pkl')
shap_values_all = pickle.load(open(shap_path, "rb"))

@st.cache_data #mise en cache de la fonction
def load_data():
    data_path = os.path.join('Data', 'X_test.csv')
    df = pd.read_csv(data_path, index_col=0)
    return df

X_test = load_data()
feature_names = X_test.columns.tolist()

# Titre 1
st.markdown("""
            <h1>
            9. Quelles sont les variables globalement les plus importantes pour comprendre la prédiction ?
            </h1>
            """, 
            unsafe_allow_html=True)
st.write("")

st.write("""
         **Rappel sur l'interprétabilité d'un modèle**
         
         L’interprétabilité correspond à la mesure dans laquelle un être humain peut prédire de manière cohérente
         le résultat d’un modèle. Plus l'interprétabilité d'un modèle de Machine Learning est élevée,
         plus il est facile pour un individu de comprendre le raisonnement derrière certaines décisions ou prédictions.
         Un modèle est plus facilement interprétable qu'un autre si ses décisions sont plus faciles à comprendre pour un humain.
    """)

st.write("""
         **Interprétation de modèle avec SHAP**
         
         SHAP pour *SHapley Additive exPlanations* est une librairie Python proposant une approche unifiée pour
         expliquer le résultat de tout modèle de Machine Learning. Elle utilise la théorie des jeux pour attribuer
         une importance à chaque variable (feature) d'entrée en mesurant la contribution marginale de chaque feature
         à la prédiction finale.
         
         SHAP permet de répondre à la question : **« Pourquoi le modèle a-t-il prédit cette valeur ? »**.
         Elle fournit ainsi des explications claires et compréhensibles pour les décideurs et les utilisateurs finaux,
         ce qui renforce la transparence et la confiance dans les prédictions du modèle.
    """)

st.write("""
         L’importance des variables est calculée en moyennant la valeur absolue des valeurs de Shap.
         Les caractéristiques sont classées de l'effet le plus élevé au plus faible sur la prédiction.
         Le calcul prend en compte la valeur SHAP absolue, donc peu importe si la fonctionnalité affecte
         la prédiction de manière positive ou négative.
    """)

st.write("""
         *__Le diagramme d'importance des variables__* répertorie les variables les plus significatives par ordre décroissant.
        Les *__variables en haut__* contribuent davantage au modèle que celles en bas et ont donc un *__pouvoir prédictif élevé__*.
        
        Puisque nous traitons une tâche de classification multi-classes, le tracé récapitulatif classe les variables
        en fonction de leur contribution globale à toutes les classes et code en couleur l'ampleur de chaque classe.
    """)

fig = plt.figure()
shap.summary_plot(shap_values_all,
                X_test,
                plot_type="bar",
                class_names=target_names,
                feature_names=feature_names,
                plot_size=(16, 12),
                max_display=33,
                show=False)
plt.title(
    "Importance des features dans la construction du modèle LGBM multi-classes",
    fontsize=20,
    fontstyle='italic')
plt.tight_layout()
st.pyplot(fig)

st.write("***Ainsi, la présence ou non de la ceinture lors de l'accident est la variable qui contribue le plus à la performance \
         du modèle LGBM multi-classes et surtout pour la prédiction des personnes 'indemnes'.***")

# Titre
st.markdown("""
            <h1>
            10. Quel est l'Impact de chaque caractéristique sur la prédiction de chaque classe ?
            </h1>
            """, 
            unsafe_allow_html=True)
st.write("")

st.write("Les diagrammes des valeurs SHAP ci-dessous indique également comment chaque caractéristique impacte la prédiction. \
        Les valeurs de Shap sont représentées pour chaque variable dans leur ordre d’importance. \
        Chaque point représente une valeur de Shap (c'est-à-dire un usager accidenté de la route.")
st.write("")

fig = plt.figure()
ax0 = fig.add_subplot(221)
shap.summary_plot(shap_values_all[0], 
                  features=X_test,
                  feature_names=feature_names,
                  class_names= target_names,
                  cmap='PiYG_r',
                  plot_type="dot",
                  max_display=15,
                  show = False)
plt.title(f"Interprétation Globale de la classe {target_names[0]}\n", 
          fontsize=20, fontstyle='italic', fontweight='bold')

ax1 = fig.add_subplot(222)
shap.summary_plot(shap_values_all[1], 
                  features=X_test,
                  feature_names=feature_names,
                  class_names= target_names,
                  cmap='PiYG_r',
                  plot_type="dot",
                  max_display=15,
                  show = False)
plt.title(f"Interprétation Globale de la classe {target_names[1]}\n", 
          fontsize=20, fontstyle='italic', fontweight='bold')

ax2 = fig.add_subplot(223)
shap.summary_plot(shap_values_all[2], 
                  features=X_test,
                  feature_names=feature_names,
                  class_names= target_names,
                  cmap='PiYG_r',
                  plot_type="dot",
                  max_display=15,
                  show = False)
plt.title(f"\nInterprétation Globale de la classe {target_names[2]}\n", 
          fontsize=20, fontstyle='italic', fontweight='bold')

ax3 = fig.add_subplot(224)
shap.summary_plot(shap_values_all[3], 
                  features=X_test,
                  feature_names=feature_names,
                  class_names= target_names,
                  cmap='PiYG_r',
                  plot_type="dot",
                  max_display=15,
                  show = False)
plt.title(f"\nInterprétation Globale de la classe {target_names[3]}\n", 
          fontsize=20, fontstyle='italic', fontweight='bold')
plt.gcf().set_size_inches(24,16)
plt.tight_layout() 
st.pyplot(plt.gcf())

plt.clf()  # Nettoyez la figure courante


# Titre 
st.markdown("""
            <h1>
            11. Interprétation des prédictions par classe
            </h1>
            """, 
            unsafe_allow_html=True)
st.write("""
         Nous pouvons obtenir un aperçu plus approfondi de l'effet de chaque feature
         pour chaque classe sur l'ensemble de données avec un graphique de dépendance.
    """)


# Création et affichage du sélecteur des variables et des graphs de dépendance
feature_names = sorted(feature_names)
col1, col2, = st.columns(2) # division de la largeur de la page en 2 pour diminuer la taille du menu déroulant
with col1:
    ID_var = st.selectbox("*Veuillez sélectionner une variable à l'aide du menu déroulant 👇*", 
                            (feature_names), index=0)
    st.write("Vous avez sélectionné la variable :", ID_var)

st.write(f"Les catégories de la variable **{ID_var}** sont sur l'axe des abscisses.")

st.write("""
         Pour chaque classe, les valeurs de SHAP les plus élevées indiquent les catégories de la variable d'entrée
         qui ont le plus grand impact sur les prédictions du modèle.
    """)

fig = plt.figure()
ax0 = fig.add_subplot(222)
shap.dependence_plot(ID_var, 
                     shap_values_all[0], 
                     X_test, 
                     interaction_index=None,
                     alpha=0.5,
                     ax=ax0, 
                     show = False)
plt.title(f"Graphique de dépendance\nVariable : '{ID_var}'\nClasse : '{target_names[0]}'",
          fontsize=20,
          fontstyle='italic')

ax1 = fig.add_subplot(223)
shap.dependence_plot(ID_var, 
                     shap_values_all[1], 
                     X_test, 
                     interaction_index=None, 
                     alpha=0.5,
                     ax=ax1, show = False)
plt.title(f"Graphique de dépendance\nVariable : '{ID_var}'\nClasse : '{target_names[1]}'",
          fontsize=20,
          fontstyle='italic')

ax2 = fig.add_subplot(224)
shap.dependence_plot(ID_var, 
                     shap_values_all[2], 
                     X_test, 
                     interaction_index=None,
                     alpha=0.5,
                     ax=ax2, show = False)
plt.title(f"Graphique de dépendance\nVariable : '{ID_var}'\nClasse : '{target_names[2]}'",
          fontsize=20,
          fontstyle='italic')

ax3 = fig.add_subplot(221)
shap.dependence_plot(ID_var, 
                     shap_values_all[3], 
                     X_test, 
                     interaction_index=None, 
                     alpha=0.5,
                     ax=ax3, show = False)
plt.title(f"Graphique de dépendance\nVariable : '{ID_var}'\nClasse : '{target_names[3]}'",
          fontsize=20,
          fontstyle='italic')

plt.gcf().set_size_inches(24,16)
plt.tight_layout()
st.pyplot(plt.gcf())

plt.clf()  # Nettoyez la figure courante


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

