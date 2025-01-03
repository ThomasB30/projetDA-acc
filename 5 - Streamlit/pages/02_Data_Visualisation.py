import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FixedFormatter, FixedLocator
import plotly.express as px
import plotly.graph_objs as go
from PIL import Image
from xplotter.insights import plot_countplot
import os

def reassembler_fichier(fichier_dest, morceaux):
    with open(fichier_dest, 'wb') as destination:
        for morceau in morceaux:
            with open(morceau, 'rb') as part:
                destination.write(part.read())

morceaux = ['Data/df_accidents_renamed.csv.part0', 'Data/df_accidents_renamed.csv.part1', 'Data/df_accidents_renamed.csv.part2']
reassembler_fichier('Data/df_accidents_renamed.csv', morceaux)

# Configuration de la page
st.set_page_config(
        page_title='Visualisation et Analyse des données',
        layout="wide" )


# Lecture du fichier
@st.cache_data  # mise en cache de la fonction
def load_data():
    data_path = os.path.join('Data', 'df_accidents_renamed.csv')
    df = pd.read_csv(data_path)
    return df

df_accidents = load_data()

st.markdown("""
            <h1>
            4. Visualisation Géographique de chaque accident en France Métropolitaine.
            </h1>
            """, 
            unsafe_allow_html=True)

dpt_dom_tom = ['971', '972', '973', '974', '975', '976', '977', '978',
               '986', '987','988']

accidents_metro = df_accidents[~df_accidents['dep'].isin(dpt_dom_tom)]
accidents_metro = accidents_metro[(accidents_metro['lat']!=0.0) & (accidents_metro['long']!=0.0)].dropna(subset=['lat', 'long'], axis=0)
accidents_metro = accidents_metro[(accidents_metro['lat']!=11) & (accidents_metro['long']!=11)].dropna(subset=['lat', 'long'], axis=0)
accidents_metro = accidents_metro[~accidents_metro['gravité'].isin(['1'])]

mapbox_token = "pk.eyJ1IjoiaXNhLWNyZWEiLCJhIjoiY2xmcjBjcHQ0MDN3czNzcDE2eWVra21hMSJ9.Rx-IQFKrlCsAr8Ee6fcsHw"

# Configurer Plotly pour utiliser Mapbox
px.set_mapbox_access_token(mapbox_token)

fig = px.scatter_mapbox(accidents_metro,
                        lat='lat',
                        lon='long',
                        color='gravité',
                        zoom=5,
                        hover_name='gravité',
                        center={"lat": 46.603354, "lon": 1.888334},
                        size_max=0.3,  # Réduire la taille des points
                        opacity=0.5)  # Ajuster l'opacité des points)

fig.update_layout(
    mapbox_style="streets",
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    width=1100,  # Ajuster la largeur de la carte
    height=750,  # Ajuster la hauteur de la carte
)

# Ajuster la taille et la position de la légende
fig.update_layout(legend=dict(font=dict(size=16), 
                              title="Gravité des accidents",
                              title_font=dict(size=18), 
                              x=0, y=1.1, xanchor="left", yanchor="top",
                              orientation="h"))

st.plotly_chart(fig)

st.markdown("""
            Des accidents mortels, il y en a partout en France.
            """)



# liste des variables
liste_variables = df_accidents.columns.to_list()
liste_variables = sorted(liste_variables)

st.markdown("""
            <h1>
            5. Répartition des usagers de la route
            </h1>
            """, 
            unsafe_allow_html=True)
st.markdown("""
            Sélectionnez 2 variables pour visualiser leur répartion et leur croisement.
            """)
st.write("")

col1, col2 = st.columns(2)
with col1:
    ID_var1 = st.selectbox("*Sélectionnez une première variable à l'aide du menu déroulant*", 
                           (liste_variables), index=9)
    st.write(f"Vous avez sélectionné la variable : *'{ID_var1}'*")
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    plot_countplot(df=df_accidents,
                   col=ID_var1,
                   order=True,
                   palette=['#8dc5bd'],
                   ax=ax1, orient='h',
                   size_labels=10)
    plt.title(f"Répartition des personnes accidentées\n en fonction de la variable {ID_var1}\n",
              loc="center", fontsize=14, fontstyle='italic', fontweight='bold', color="#5e5c5e")
    plt.grid(False)
    fig1.tight_layout()
    st.pyplot(fig1)
    
with col2:
    ID_var2 = st.selectbox("*Sélectionnez une seconde variable à l'aide du menu déroulant*", 
                           (liste_variables), index=19)
    st.write(f"Vous avez sélectionné la variable : *'{ID_var2}'*")
    
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    plot_countplot(df=df_accidents,
                   col=ID_var2,
                   order=True,
                   palette=['#b99256'],
                   ax=ax2, orient='h',
                   size_labels=10)
    plt.title(f"Répartition des personnes accidentées\n en fonction de la variable {ID_var2}\n",
              loc="center", fontsize=14, fontstyle='italic', fontweight='bold', color="#5e5c5e")
    plt.grid(False)
    fig2.tight_layout()
    st.pyplot(fig2)
    

unique_modalities = df_accidents[ID_var1].nunique()  # Calculer le nombre de modalités uniques pour ID_var1
sns.set_palette("BrBG_r", n_colors=unique_modalities)  # Utiliser n_colors pour définir le nombre de couleurs)

df_tab_croisee = pd.crosstab(df_accidents[ID_var2],
                             df_accidents[ID_var1],
                             normalize='index', margins=True, margins_name='All') * 100
fig3 = Figure(figsize=(12, 6))
ax3 = fig3.subplots()
ax3 = df_tab_croisee.plot.barh(stacked=True, rot=0, ax=ax3)
for rec in ax3.patches:
    height = rec.get_height()
    ax3.text(rec.get_x() + rec.get_width() / 2,
             rec.get_y() + height / 2,
             "{:.1f}%".format(rec.get_width()),
             ha='center',
             va='center',
             color='black',
             fontweight='bold',
             fontsize=9)
ax3.set_xlabel(f"% Répartition {ID_var1}", fontsize=10)
ax3.xaxis.set_major_locator(FixedLocator(ax3.get_xticks()))
ax3.xaxis.set_major_formatter(FixedFormatter([f"{x:.0f}" for x in ax3.get_xticks()]))
ax3.tick_params(axis='x', labelsize=10)
ax3.set_ylabel(f"{ID_var2}", fontsize=10)
ax3.yaxis.set_major_locator(FixedLocator(ax3.get_yticks()))
ax3.yaxis.set_major_formatter(FixedFormatter([y.get_text() for y in ax3.get_yticklabels()]))
ax3.tick_params(axis='y', labelsize=10)
ax3.get_legend().remove()
handles, labels = ax3.get_legend_handles_labels()
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), 
           ncol=len(labels), fontsize=12)
ax3.set_title(f"Répartition {ID_var1} par {ID_var2}",
              y=1.15, fontsize=16, fontstyle='italic', fontweight='bold', color="#5e5c5e")
st.pyplot(fig3)


st.markdown("""
            <h1>
            6. Analyse de la temporalité des accidents de la route
            </h1>
            """, 
            unsafe_allow_html=True)

df_accidents['date'] = pd.to_datetime(df_accidents['date'])
values = df_accidents['date'].value_counts().sort_index()

# Créer un nouveau DataFrame avec les données originales et la moyenne mobile
df_plot = pd.DataFrame({'date': values.index, 'value': values.values})
df_plot['rolling_mean'] = df_plot['value'].rolling(window=30).mean()


# Créer le graphique de base avec Plotly Express
fig = go.Figure()

# Ajouter la courbe du nombre d'accidents par jour
fig.add_scatter(x=df_plot['date'], y=df_plot['value'], mode='lines', name="Nb d'accidents par jour",
                line=dict(color='#9ebeb8'))

# Ajouter la courbe de la moyenne mobile
fig.add_scatter(x=df_plot['date'], y=df_plot['rolling_mean'], mode='lines', name='Moyenne Mobile 30 jours',
                line=dict(dash='dash', width=3, color="#ad7d67"))

# Personnaliser le graphique
fig.update_layout(
    title=f"Nombre d'usagers accidentés du {df_accidents.date.min().strftime('%d %b %Y')} au {df_accidents.date.max().strftime('%d %b %Y')}",
    title_xanchor='center',
    title_x=0.5,
    width=1100, height=500, template='plotly_white',
    font=dict(size=12),
    title_font=dict(size=24, color="#ad7d67"),
    xaxis=dict(tickformat="%d %b %Y", tickangle=45, tickmode="auto"),
    yaxis=dict(title="Nb d'accidents par jour"),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        title=None,
        font=dict(size=16, color="#5e5c5e")
    )
)

# Afficher le graphique dans Streamlit
st.plotly_chart(fig)

st.markdown("""
            L'année 2020 est très différente des 2 autres années. 
            Alors qu'en 2019 et 2021, le nombre d'accidents est réparti de manière homogène sur les 12 mois de l'année, 
            on constate que ce n'est pas le cas en 2020.
            
            Il y a eu nettement moins d'accidents en avril, mai et novembre, en raison des confinements liés au Covid-19.
            """)


st.markdown("""
            <h2>
            Choisissez la modalité de la variable dont vous souhaitez analyser la temporalité
            </h2>
            """, 
            unsafe_allow_html=True)

cols = st.columns(2)
variable_choice = cols[0].selectbox("Choisissez la variable :", options=liste_variables, index=1)
modalite_choice = cols[1].selectbox("Choisissez la modalité :", options=sorted(df_accidents[variable_choice].unique()), index=1)

df_accidents_filtered = df_accidents[df_accidents[variable_choice] == modalite_choice]

values = df_accidents_filtered['date'].value_counts().sort_index()

# Créer un nouveau DataFrame avec les données filtrées et la moyenne mobile
df_plot = pd.DataFrame({'date': values.index, 'value': values.values})
df_plot['rolling_mean'] = df_plot['value'].rolling(window=30).mean()

# Créer le graphique de base avec Plotly Express
fig = go.Figure()

# Ajouter la courbe du nombre d'accidents par jour
fig.add_scatter(x=df_plot['date'], y=df_plot['value'], mode='lines', name="Nb d'accidents par jour",
                line=dict(color='#9ebeb8'))

# Ajouter la courbe de la moyenne mobile
fig.add_scatter(x=df_plot['date'], y=df_plot['rolling_mean'], mode='lines', name='Moyenne Mobile 30 jours',
                line=dict(dash='dash', width=3, color="#ad7d67"))

# Personnaliser le graphique
fig.update_layout(
    title=f"Evolution de nombre de '{modalite_choice}' du {df_accidents.date.min().strftime('%d %b %Y')} au {df_accidents.date.max().strftime('%d %b %Y')}",
    title_xanchor='center',
    title_x=0.5,
    width=1100, height=500, template='plotly_white',
    font=dict(size=12),
    title_font=dict(size=24, color="#ad7d67"),
    xaxis=dict(tickformat="%d %b %Y", tickangle=45, tickmode="auto"),
    yaxis=dict(title="Nb d'accidents par jour"),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        title=None,
        font=dict(size=16, color="#5e5c5e")
    )
)

# Afficher le graphique dans Streamlit
st.plotly_chart(fig)


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

