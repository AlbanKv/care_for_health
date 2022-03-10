import json
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
import json

st.set_page_config(
     page_title="General practitioners repartition",
     page_icon="🐍",
     layout="wide",
     initial_sidebar_state="expanded",
    )

st.markdown('''
# General Practitioners repartition
#### Improving daily access to general practitioners in France, based on a different repartition.''')

columns = st.columns(3)

radius = columns[0].text_input('Select radius', value='15')
breakeven = columns[1].text_input('Set a breakeven ratio', value='0.84')
poids_des_voisins = columns[2].text_input('Neighbors_weight', value='0.1')

columns2 = st.columns(3)

med_pickup = columns2[0].selectbox(
     'How would you pick general practitioners?',
     ('As many as possible', 'Only where they are too numerous'))
how_to_sort = columns2[1].selectbox(
     'Which type of spread do you want to choose?',
     ('Distance_based', 'Calcul_based'))
nb_voisins_minimum = columns2[2].slider('Minimum number of neighbors', 1, 10, 3)#selectbox('Minimum number of neighbors', range(1,11))

if med_pickup=='As many as possible':
    med_pickup='tous'
else:
    med_pickup='excédent'

if how_to_sort=='Distance_based':
    sortby='distance'
else:
    sortby='calculated'

url = 'https://careforhealth-gfiqg24vta-ew.a.run.app/predict'
#url='http://127.0.0.1:8000/predict'#local

params=dict(
    selection_medecins=str(med_pickup),
    sortby=sortby,
    moy_region=str(breakeven),
    radius=radius,
    recalcul=False,
    )

'''
    poids_des_voisins=poids_des_voisins,
    nb_voisins_minimum=nb_voisins_minimum,
'''
df_default = pd.read_csv('data/df_api_test.csv', delimiter=',', dtype={'code_insee':'str'}, usecols=['code_insee', 'neighbors_taux_de_couverture']).reindex()# converters={"neighbors": lambda x: ast.literal_eval(x)}).reindex()
df_combine = df_default
st.markdown('''


''')
dicty={}

if st.button('Make the magic happen'):
    # print is visible in the server output, not in the page
    print('You made it!')
    req = requests.get(url, params=params)
    dicty = json.loads(req.json()['data'])
    st.markdown(f'''
        ## {req.json()['Nouveau_taux']:.2f}
        ''')
    columns_res = st.columns(2)
    columns_res[0].write(f"Initial rate :{req.json()['Ancien_taux']:.2f}")
    columns_res[1].write('test2')#f'Rate progression: {100*(req.json()['Evolution du taux']):.2f}')

    df_from_dicty = pd.DataFrame(dicty).reset_index().rename(columns={'index': 'code_insee'})
    df_combine=df_default.drop(columns='neighbors_taux_de_couverture').merge(df_from_dicty, how='left', left_on='code_insee', right_on='code_insee')

else:
    st.write('Click me!')

#st.dataframe(data=pd.DataFrame(dicty))

# @st.cache
def chloropleth_map_communes(df_communes,code_insee_str,taux_couv_str):
    """
    arguments :
        code_insee_str: mettre le nom de la colonne ou le code INSEE est présent ('code_insee')
        taux_couv_str: mettre le nom de la colonne ou le taux de couverture est présent ('taux_de_couverture')
    """
    #json_load
    json_data = 'raw_data/communes_fr.json'
    json_load = json.load(open(json_data))

    #map
    fig = go.Figure(go.Choroplethmapbox(
            geojson = json_load, #Assign geojson file : délimitation des régions
            featureidkey = "properties.codgeo", #Assign feature key : code INSEE
            locations = df_communes[code_insee_str], #Assign location data : code INSEE
            z = df_communes[taux_couv_str], #Assign information data : taux de couverture
            zmin=0, zmax=1.5,
            colorscale = [[0, 'rgb(0,0,255)'], [0.5, 'rgb(0,255,0)'], [1, 'rgb(255,0,0)']],
            showscale = True))

    fig.update_layout(
        width = 600,
        height = 600,
        mapbox_style = "carto-positron",
        mapbox_zoom = 4,
        mapbox_center = {"lat": 46.227638, "lon": 2.213749}, #Centre de la France
    )

    return fig

#fig.update_layout(mapbox_style="stamen-terrain", mapbox_zoom=10, mapbox_center_lat = mid_lat, mapbox_center_lon = mid_lon,
#    margin={"r":0,"t":0,"l":0,"b":0})

#@st.cache
def display_chloro_cache2():
    fig = chloropleth_map_communes(df_combine,'code_insee', 'neighbors_taux_de_couverture')
    return fig

fig = display_chloro_cache2()
st.plotly_chart(fig, use_container_width=True)

#st.plotly_chart(fig, use_container_width=True)

#fig.update_traces(
#        z = df_combine['neighbors_taux_de_couverture'], #Assign information data : taux de couverture
#        zmin=0, zmax=1.5,
#        colorscale = [[0, 'rgb(0,0,255)'], [0.6, 'rgb(0,255,0)'], [1, 'rgb(255,0,0)']],
#)
#fig.update_layout()
#return chloropleth_map_communes(df_combine,'code_insee', 'neighbors_taux_de_couverture')


#st.plotly_chart(fig, use_container_width=True)




#fig = chloropleth_map_communes(df_communes, code_insee_str, taux_couv_str)

## MAP PAR COMMUNES (SELECTION DES REGIONS A FAIRE DANS LE DATAFRAME)


# st.write(req.get('fare'))
'''
# @st.cache
def plot_map():
    return px.line_mapbox(my_trip, lat='lat', lon='lon', zoom=3, height=400)
    
px.scatter_mapbox(my_trip, lat='lat', lon='lon')

fig = plot_map()

fig.update_layout(mapbox_style="stamen-terrain", mapbox_zoom=10, mapbox_center_lat = mid_lat, mapbox_center_lon = mid_lon,
    margin={"r":0,"t":0,"l":0,"b":0})

st.plotly_chart(fig, use_container_width=True)

fig.update_layout()

'''