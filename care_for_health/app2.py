from turtle import fillcolor
import streamlit as st
import leafmap.foliumap as leafmap
import pandas as pd 
import requests
import json

st.set_page_config(
     page_title="General practitioners repartition",
     page_icon="",
     layout="wide",
     initial_sidebar_state="expanded",
    )

st.markdown('''
# General Practitioners repartition
#### Improving daily access to general practitioners in France, based on a different repartition.''')

columns = st.columns(3)

radius = columns[0].slider('Select radius (in Km)', 5, 50, 15)#text_input('Select radius', value='15')
        
med_pickup = columns[1].selectbox('How would you pick general practitioners?',
     ('As many as possible', 'Only where they are too numerous'))
how_to_sort = columns[2].selectbox('Which type of spread do you want to choose?',
     ('The nearest neighbors first', 'Calcul based'))

columns2 = st.columns(3)

expander = st.expander("Optional controls")
columns3 = expander.columns(3)
breakeven = columns3[0].text_input('Set a breakeven ratio', value='None')
weight = columns3[1].text_input('Neighbors_weight', value='0.1')
nb_neighbors = columns3[2].slider('Minimum number of neighbors', 1, 10, 3)#selectbox('Minimum number of neighbors', range(1,11))

if med_pickup=='As many as possible':
    med_pickup='tous'
else:
    med_pickup='excédent'

if how_to_sort=='Distance_based':
    sortby='distance'
else:
    sortby='calculated'

params=dict(
    selection_medecins=str(med_pickup),
    sortby=sortby,
    moy_region=str(breakeven),
    radius=str(radius),
    )
    #recalcul=False,
    #poids_des_voisins=str(weight),
    #nb_voisins_minimum=str(nb_neighbors),

url = 'https://careforhealth-gfiqg24vta-ew.a.run.app/predict'

apps = {
    "heatmap": {'title': "Heatmap", "icon": "map"}
}

df_default = pd.read_csv('data/df_api_test.csv', delimiter=',', dtype={'code_insee':'str'}, usecols=['code_insee', 'neighbors_taux_de_couverture']).reindex()# converters={"neighbors": lambda x: ast.literal_eval(x)}).reindex()
df_combine = df_default
dicty={}



filepath = "brouillon/df_api_test.csv"#https://raw.githubusercontent.com/giswqs/leafmap/master/examples/data/us_cities.csv"
df = pd.read_csv(filepath, usecols=['code_insee', 'neighbors_taux_de_couverture', 'Lat_commune', 'Lon_commune'])

lowest = df[df['neighbors_taux_de_couverture']<=0.6]
low_mid = df[(df['neighbors_taux_de_couverture']>0.6)&(df['neighbors_taux_de_couverture']<=0.8)]
mid = df[(df['neighbors_taux_de_couverture']>0.8)&(df['neighbors_taux_de_couverture']<=1.0)]
high_mid = df[(df['neighbors_taux_de_couverture']>1.0)&(df['neighbors_taux_de_couverture']<=1.2)]
highest = df[df['neighbors_taux_de_couverture']>1.2]

requested = 1
# Requête API
if requested==1:
    if st.button('Make the magic happen'):
        print(requested)
        req = requests.get(url, params=params)
        dicty = json.loads(req.json()['data'])
        st.markdown(f'''
            ## {req.json()['Nouveau_taux']:.2f}
            ''')
        columns_res = st.columns(2)
        columns_res[0].write(f"Initial rate :{req.json()['Ancien_taux']:.2f}")
        columns_res[1].write('test2')#f'Rate progression: {100*(req.json()['Evolution du taux']):.2f}')

        df_from_dicty = pd.DataFrame(dicty).reset_index().rename(columns={'index': 'code_insee'})
        #df_combine=df_default.drop(columns='neighbors_taux_de_couverture').merge(df_from_dicty, how='left', left_on='code_insee', right_on='code_insee')


def heatmap():
    filepath = "brouillon/df_api_test.csv"#https://raw.githubusercontent.com/giswqs/leafmap/master/examples/data/us_cities.csv"
    m = leafmap.Map(tiles="openstreetmap", center=(df['Lat_commune'].mean(), df['Lon_commune'].mean()), draw_export=True, zoom=8)
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