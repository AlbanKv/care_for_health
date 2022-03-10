from turtle import fillcolor
import streamlit as st
import leafmap.foliumap as leafmap
import pandas as pd 

st.set_page_config(
     page_title="General practitioners repartition",
     page_icon="",
     layout="wide",
     #initial_sidebar_state="expanded",
    )

st.markdown('''
# General Practitioners repartition
#### How could we improve daily access to a general practitioner in France, based only on a different repartition? 
''')

st.markdown('''
#### Set parameters:
''')









apps = {
    "heatmap": {'title': "Heatmap", "icon": "map"}
}



filepath = "brouillon/df_api_test.csv"#https://raw.githubusercontent.com/giswqs/leafmap/master/examples/data/us_cities.csv"
df = pd.read_csv(filepath)
lowest = df[df['neighbors_taux_de_couverture']<=0.6]
low_mid = df[(df['neighbors_taux_de_couverture']>=0.6)&(df['neighbors_taux_de_couverture']<=0.8)]
mid = df[(df['neighbors_taux_de_couverture']>=0.8)&(df['neighbors_taux_de_couverture']<=1.0)]
high_mid = df[(df['neighbors_taux_de_couverture']>=1.0)&(df['neighbors_taux_de_couverture']<=1.2)]
highest = df[df['neighbors_taux_de_couverture']>=1.2]

def heatmap():

    gdf="raw_data/communes_fr.json"

    filepath = "brouillon/df_api_test.csv"#https://raw.githubusercontent.com/giswqs/leafmap/master/examples/data/us_cities.csv"
    m = leafmap.Map(tiles="openstreetmap", center=(47, -0.5), draw_export=True, zoom=8)
    m. add_circle_markers_from_xy(
        data=highest, 
        x='Lon_commune', 
        y='Lat_commune', 
        z='neighbors_taux_de_couverture',
        radius=10, 
        popup='code_insee', 
        tooltip=None, 
        min_width=100, 
        opacity=1.0,
        max_width=200, 
        color="#00c3ff",
        fill_color='#00c3ff',
        stroke=False,
        )
    m. add_circle_markers_from_xy(
        data=high_mid, 
        x='Lon_commune', 
        y='Lat_commune', 
        z='neighbors_taux_de_couverture',
        radius=10, 
        popup=None, 
        tooltip=None, 
        min_width=100, 
        max_width=200, 
        color="#00ffe5",
        fill_color='#00ffe5',
        stroke=False,
        )
    m. add_circle_markers_from_xy(
        data=mid, 
        x='Lon_commune', 
        y='Lat_commune', 
        z='neighbors_taux_de_couverture',
        radius=10, 
        popup=None, 
        tooltip=None, 
        min_width=100, 
        max_width=200, 
        color="#00ff62",
        fill_color='#00ff62',
        stroke=False,
        )
    m. add_circle_markers_from_xy(
        data=low_mid, 
        x='Lon_commune', 
        y='Lat_commune', 
        z='neighbors_taux_de_couverture',
        radius=10, 
        popup=None, 
        tooltip=None, 
        min_width=100, 
        max_width=200, 
        color="#ffdd00",
        fill_color='#ffdd00',
        stroke=False,
        )
    m. add_circle_markers_from_xy(
        data=lowest, 
        x='Lon_commune', 
        y='Lat_commune', 
        z='neighbors_taux_de_couverture',
        radius=10, 
        popup=None, 
        tooltip=None, 
        min_width=100, 
        max_width=200, 
        color="#ff6a00",
        fill_color='#ff6a00',
        stroke=False,
        )
    m.to_streamlit(height=700)

for app in apps:
    if apps[app]["title"] == 'Heatmap':
        eval(f"{heatmap()}")
        break