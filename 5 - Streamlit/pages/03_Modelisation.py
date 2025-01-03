import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from xplotter.insights import plot_countplot
from PIL import Image
import pickle
import os


# Configuration de la page
st.set_page_config(
        page_title='Prédiction de la Gravité des Accidents de la Route',
        layout="wide" )

st.markdown("""
            <h1>
            6. Prédiction de la Gravité des Accidents de la Route
            </h1>
            """, 
            unsafe_allow_html=True)

st.markdown("""
            Notre étude a pour objectif de modéliser la gravité des accidents de la route en France
            en se concentrant sur quatre classes de gravité : indemnes, blessés légers, hospitalisés et tués.
            La gravité représente la variable cible du projet et comme nous pouvons le voir, celle-ci est composée
            de quatre classe. Nous sommes donc en présence d'un problème de classification multiclasse.
            
            Nous avons décomposé notre jeu de données en 2 sous-ensembles : 80% pour entraîner nos modèles de prédiction
            et 20% pour les tester. L’utilisation de la fonction « train_test_split » de la librairie sklearn permet
            de diviser le jeu de données de manière à maintenir la même proportion de chaque classe dans les ensembles
            d'apprentissage et de test. Cette approche est particulièrement utile lorsque certaines classes sont moins
            représentées que d'autres, car elle garantit que les sous-ensembles contiennent suffisamment d'exemples de chaque
            classe pour une évaluation précise du modèle.
            """)


# Lecture et préparation des data
@st.cache_data  # mise en cache de la fonction
def load_X_train():
    data_path = os.path.join('Data', 'X_train.csv')
    df = pd.read_csv(data_path, index_col=0)
    return df
X_train = load_X_train()

@st.cache_data  # mise en cache de la fonction
def load_X_test():
    data_path = os.path.join('Data', 'X_test.csv')
    df = pd.read_csv(data_path, index_col=0)
    return df
X_test = load_X_test()

@st.cache_data  # mise en cache de la fonction
def load_y_train():
    data_path = os.path.join('Data', 'y_train.csv')
    df = pd.read_csv(data_path, index_col=0)
    # df = df['grav']
    df = pd.DataFrame(df, columns=['grav'])
    return df
y_train = load_y_train()

@st.cache_data  # mise en cache de la fonction
def load_y_test():
    data_path = os.path.join('Data', 'y_test.csv')
    df = pd.read_csv(data_path, index_col=0)
    # df = df['grav']
    df = pd.DataFrame(df, columns=['grav'])
    return df
y_test = load_y_test()


# Répartition de la variable cible dans les 2 jeux de données
fig = plt.figure(figsize=(20, 6))
ax1 = fig.add_subplot(1,2,1)
plot_countplot(df=y_train,
               col='grav',
               order=True,
               palette=['#d4b3ac'],
               ax=ax1, orient='v',
               size_labels=12)
plt.grid(visible=False)
plt.title("Répartition des usagers selon la gravité de leur accident\n dans l'ensemble d'entraînement",
          loc="center", fontsize=16, fontstyle='italic', fontweight='bold', color="#5e5c5e")
ax2 = fig.add_subplot(1,2,2)
plot_countplot(df=y_test,
               col='grav',
               order=True,
               palette=['#d4b3ac'],
               ax=ax2, orient='v',
               size_labels=12)
plt.grid(visible=False)
plt.title("Répartition des usagers selon la gravité de leur accident\n dans l'ensemble de test",
          loc="center", fontsize=16, fontstyle='italic', fontweight='bold', color="#5e5c5e")
plt.grid(False)
fig.tight_layout()
st.pyplot(fig)

st.write("")
st.markdown("""
            Comme nous pouvons le voir, la variable cible présente une répartition hétéorgène entre ses classes.
            
            **Après voir testé plusieurs modèles de prédiction pour notre classification déséquilibrée, 
            nous avons sélectionné les trois modèles présentant les meilleures performances :**
              - Le RandomForest Classifier ;
              - Le LGBM Classifier ;
              - Le GradientBoosting Classifier.
            
            Sur ces trois modèles, nous avons réalisé une recherche des meilleurs hyperparamètres selon
            la méthodologie suivante :
              - Utilisation du RandomSearch afin d’obtenir un ordre de grandeur sur les différents hyperparamètres ;
              - Utilisation du GridSearch pour affiner les hyperparamètres autour des valeurs trouvées précédemment.
            
            Un resampling des données d'entrainement a également été réalisé. L'objectif est de réduire les différences
            entre la répartition des classes (en augmentant le nombre d'échantillons des classes minoritaire et en réduisant
            le nombre d'échantillons des classes majoritaires).
                
            Les résultats obtenus sont présentés ci-dessous.
            """)
            
image_path = os.path.join('Photo', 'Résultats 3 modèles multiclasses.png')
choix_model = Image.open(image_path)
st.image(choix_model, use_column_width="always")

st.write("")
st.markdown("""
            **Decription des modèles :**
               - **RandomForestClassifier :** Ce modèle utilise plusieurs arbres de décision pour améliorer
                 la précision des prédictions. Il peut être utilisé pour des problèmes de classification binaire
                 et multi-classes. Cependant, il peut être difficile à interpréter.
               - **LGBMClassifier :** Ce modèle utilise un algorithme de gradient boosting plus rapide
                 et plus efficace que XGBClassifier. Il est également efficace pour les problèmes de classification
                 binaire et multi-classes. Cependant, il peut être sensible aux valeurs aberrantes.
               - **GradientBoostingClassifier :** Ce modèle utilise un algorithme de gradient boosting pour améliorer
                 la précision des prédictions. Il est également efficace pour les problèmes de classification binaire
                 et multi-classes. Cependant, il peut être sensible au bruit et au sur-ajustement des données.
            
            **Rappel sur les métriques :**
               - Via le Recall, on regarde le nombre de positif que le modèle a bien prédit
                  sur l’ensemble des positifs. Si le modèle prédit uniquement « positif », le Recall sera élevé ;
               - Via la Précision, on regarde le nombre de positif que le modèle a bien prédit sur l’ensemble
                  des positifs prédit. Si le modèle ne prédit jamais « positif », la précision sera élevée ;
               - Ainsi, plus le Recall est élevé et plus le modèle repère de positif.
                  Plus la précision est élevée et moins le modèle ne se trompe sur les positifs ;
               - Avec une modélisation multiclasse, les métriques sont moyennées suivant les classes.

            De manière synthétique, nous pouvons constater que :
               - Les scores sont assez similaires entre les modèles ;
               - Le GradientBoosting présente des performances légèrement meilleures ;
               - La précision est plus élevée que le recall. Cela signifie que les modèles prédisent plus facilement des vrais positifs ;
               - Un léger over-fitting est présent, principalement pour le RandomForest.
            """)

st.markdown("""       
            *Pour plus de précisions sur la méthodologie de sélection du "meilleur" modèle, 
            vous pouvez lire le [rapport d'étude](https://github.com/ThomasB30/projetDA-acc/tree/main/rendu)*
                                 
            **Pour la suite de cette présentation, nous utiliserons les résultats du modèle LGBM pour des raisons pratiques
            (modèle plus rapide, interprétation par SHAP plus rapide).**
            """)

st.markdown("""
            Les résultats obtenus avec le modèle LGBM optimisé s'interprètent ainsi :
            
            **Précision (66%) :** La précision pondérée est la moyenne des précisions pour chaque classe,
            pondérée par le nombre d'observations de chaque classe. Une précision de 66% signifie que lorsque
            le modèle prédit une classe, il est correct environ 66% du temps, ce qui est supérieur à une classification aléatoire.
            
            **Recall (46%) :** Le rappel pondéré est la moyenne des rappels pour chaque classe, pondérée par le nombre
            d'observations de chaque classe. Un Recall de 46% signifie que le modèle identifie correctement
            environ 46% des observations de chaque classe.
            
            **F1-score (47%) :** Le F1-score pondéré est la moyenne harmonique des précisions et rappels
            pondérés pour chaque classe. Un F1-score pondéré de 47% indique que le modèle a un équilibre convenable
            entre la précision et le rappel pour chaque classe, compte tenu du jeu de données utilisé.
            """)


# Chargement du modèle
model_path = os.path.join("Modeles", "best_model_LGBM_resampled.pkl")
model = pickle.load(open(model_path, "rb"))

st.markdown("""
            <h1>
            7. Prédisez la Gravité de l'Accident de la Route
            </h1>
            """, 
            unsafe_allow_html=True)
st.markdown("""
            Décrivez l'accident de la route en sélectionnant à l'aide des filtres ci-desous ses caractéristiques.<br>
            Puis cliquez sur le bouton "Prédiction" pour en prédire la gravité.
            """,
            unsafe_allow_html=True)


# Ce code utilise st.session_state pour gérer l'état de session.
# Les prédictions seront réinitialisées chaque fois que l'utilisateur
# modifie la sélection de modalités.

# Initialisez l'état de session si nécessaire
if "prediction_text" not in st.session_state:
    st.session_state["prediction_text"] = None

if "prob_df" not in st.session_state:
    st.session_state["prob_df"] = None

if "reset_clicked" not in st.session_state:
    st.session_state["reset_clicked"] = False

# Récupérez les variables qualitatives
qualitative_vars = X_train.columns.tolist()

# Créez un dictionnaire pour stocker les valeurs sélectionnées
selected_values = {}

# Définissez les largeurs des colonnes principales
main_col_width = 1.5

# Divisez l'espace en 4 colonnes avec des largeurs personnalisées
cols = st.columns([main_col_width, main_col_width, main_col_width, main_col_width])

# Attribuez des colonnes
col1, col2, col3, col4 = cols[0], cols[1], cols[2], cols[3]

# Utilisez une boucle pour créer les selectbox pour chaque variable qualitative
for idx, var in enumerate(qualitative_vars):
    # Récupérez les modalités uniques pour la variable en cours
    unique_values = X_train[var].unique().tolist()

    # Choisissez la colonne en fonction de l'index de la variable
    if idx < len(qualitative_vars)*0.25:
        current_col = col1
    elif (idx >= len(qualitative_vars)*0.25) and (idx < len(qualitative_vars)*0.5):
        current_col = col2
    elif (idx >= len(qualitative_vars)*0.5) and (idx < len(qualitative_vars)*0.75):
        current_col = col3
    else:
        current_col = col4

    # Créez une selectbox pour la variable en cours dans la colonne correspondante
    selected_value = current_col.selectbox(f"Modalité pour **{var}** :", unique_values)

    # Stockez la valeur sélectionnée dans le dictionnaire
    selected_values[var] = selected_value

# Affichez le dictionnaire avec les valeurs sélectionnées
st.write("Valeurs sélectionnées pour chaque variable :", selected_values)

# On crée un bouton pour réaliser les prédictions
clicked = st.button("Cliquez pour connaître la prédiction de la gravité de l'accident")

# On initialise ces variables pour pas avoir d'erreur si pas de prédictions faites
prediction = None
probabilities = None
input_data = None

# Vérifiez si le bouton est cliqué
if clicked:
    st.session_state.reset_clicked = False

    # Transformez les sélections en un DataFrame
    input_data = pd.DataFrame([selected_values])

    # Effectuez la prédiction avec le modèle
    prediction = model.predict(input_data)

    # Récupérez la prédiction sous forme de texte sans crochets
    st.session_state.prediction_text = prediction.item()
    if st.session_state.prediction_text == 1:
        st.session_state.prediction_text = 'Indemne'
    elif st.session_state.prediction_text == 2:
        st.session_state.prediction_text = 'Tué'
    elif st.session_state.prediction_text == 3:
        st.session_state.prediction_text = 'Blessé hosp.'
    else:
        st.session_state.prediction_text = 'Blessé léger'
        

    # Obtenez les probabilités pour chaque classe
    probabilities = model.predict_proba(input_data)

    # Créez un DataFrame avec les probabilités et les noms de classe
    st.session_state.prob_df = pd.DataFrame(probabilities, columns=model.classes_)

    # Trier les colonnes dans l'ordre souhaité
    st.session_state.prob_df = st.session_state.prob_df[[1, 4, 3, 2]]
    st.session_state.prob_df = st.session_state.prob_df.rename(columns={1:'Indemne', 2:'Tué', 3:'Blessé hosp.', 4:'Blessé léger'})
    st.session_state.prob_df = st.session_state.prob_df.rename(index={0:'Proba (%)'})

    # Convertissez les probabilités en pourcentage et arrondissez à 1 chiffre décimal
    st.session_state.prob_df = (st.session_state.prob_df * 100).round(1)

# Ajoutez un bouton pour effacer la prédiction et réinitialiser l'application
reset_button = st.button("Je souhaite refaire une prédiction et changer les caractéristiques de l'accident")

if reset_button:
    st.session_state.prediction_text = None
    st.session_state.prob_df = None
    st.session_state.reset_clicked = True

# Affichez le résultat de la prédiction et les probabilités si disponibles et si le bouton de réinitialisation n'a pas été cliqué
if (st.session_state.reset_clicked is False) and (st.session_state.prediction_text is not None) and (st.session_state.prob_df is not None):
    st.write(f"La prédiction de la gravité de l'accident est : **{st.session_state.prediction_text}**")
    st.write("Les probabilités pour chaque classe sont (en %) :")
    st.write(st.session_state.prob_df)


#Stockez la prédiction et les données d'entrée dans st.session_state
st.session_state.prediction = prediction
st.session_state.probabilities = probabilities
st.session_state.input_data = input_data
st.markdown("""
            ***Pour comprendre cette prédiction, rendez-vous à la page suivante en cliquant sur
            'Explication de la prédiction' dans la barre latérale gauche.***
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

