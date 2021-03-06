import streamlit as st
import leafmap.foliumap as leafmap
import pandas as pd 
import requests
import json
from sklearn.metrics import mean_absolute_error
import numpy as np 
import matplotlib.pyplot as plt
import ast
st.set_page_config(
     page_title="General practitioners repartition",
     page_icon="",
     layout="wide",
     initial_sidebar_state="expanded",
    )
region=[94]
region_dict= {
        "Auvergne-Rhône-Alpes":84,
        "Bourgogne-Franche-Comté": 27,
        "Bretagne": 53,
        "Centre-Val de Loire": 24,
        "Corse": 94,
        "Grand Est": 44,
        "Hauts-de-France": 32,
        "Île-de-France": 11,
        "Normandie": 28,
        "Nouvelle-Aquitaine": 75,
        "Occitanie": 76,
        "Pays de la Loire": 52,
        "Provence-Alpes-Côte d'Azur": 93,}

@st.cache
def get_select_box_data():
    return pd.DataFrame({
          'Region': list(region_dict.keys()),
          'Value': list(region_dict.values())
        })

reg_mini_df = get_select_box_data()


st.markdown('''
# General Practitioners repartition''')
columns0 = st.columns(2)
columns0[0].markdown('''#### Improving daily access to general practitioners in France, based on a different repartition.''')
option = columns0[1].selectbox('Select a Region', reg_mini_df['Region'], index=4)

columns = st.columns(3)

radius = columns[0].slider('Select radius (in Km)', 5, 50, 15)#text_input('Select radius', value='15')
        
med_pickup = columns[1].selectbox('How would you pick general practitioners?',
     ('Only where they are too numerous', 'As many as possible'))
how_to_sort = columns[2].selectbox('Which type of spread do you want to choose?',
     ('Nearest neighbors first', 'Worst ratio first', 'Numerous missing GPs first', 'Worst weighted ratio first', 'Combination of all of it'))
columns2 = st.columns(3)

expander = st.expander("Optional controls")
columns3 = expander.columns(3)
breakeven = columns3[0].text_input('Set a breakeven ratio', value=None)
weight = columns3[1].text_input('Neighbors_weight', value='0.1')
nb_neighbors = columns3[2].slider('Minimum number of neighbors', 1, 10, 3)#selectbox('Minimum number of neighbors', range(1,11))

if med_pickup=='As many as possible':
    selection_medecins='tous'
else:
    selection_medecins='excédent'

if how_to_sort=='Nearest neighbors first':
    sortby='distance'
elif how_to_sort=='Worst ratio first':
    sortby='deficit_rate'
elif how_to_sort=='Numerous missing GPs first':
    sortby='deficit_absolute'
elif how_to_sort=='Worst weighted ratio first':
    sortby='computed_need'
elif how_to_sort=='Combination of all of it':
    sortby='calculated'


#url = 'http://localhost:8000/predict'
url = 'https://careforhealth-gfiqg24vta-ew.a.run.app/predict'

apps = {
    "heatmap": {'title': "Heatmap", "icon": "map"}
}

df_cols = {
    'code_insee': 'str', 
    'code_regions': 'int',
    'neighbors_taux_de_couverture': 'float', 
    'Lat_commune': 'float', 
    'Lon_commune': 'float', 
    'code_regions': 'int',
    }

@st.cache
def calls_csv():
    url_df_init = 'http://localhost:8000/initialdf'
    params_init=dict(code_region=region_dict[option])
    response = requests.get(url_df_init, params=params_init)
    data_init=response.json()['data']
    return pd.DataFrame.from_dict(ast.literal_eval(data_init)).reset_index().rename(columns={'index': 'code_insee'})
    #filepath = 'data/df_api_france_9.csv'#"brouillon/df_api_test.csv"#https://raw.githubusercontent.com/giswqs/leafmap/master/examples/data/us_cities.csv"
    #return pd.read_csv(filepath, usecols=list(df_cols.keys()), dtype=df_cols)
df = calls_csv()


region_val=region_dict[option]

df_combine = df.copy()#[df['code_regions']==region_val].copy()

mae_init=mean_absolute_error(df_combine['neighbors_taux_de_couverture'], np.full(len(df_combine['neighbors_taux_de_couverture']),np.mean(df_combine['neighbors_taux_de_couverture'])))

if "mae_init" not in st.session_state:
    st.session_state.mae_init = mae_init#=mean_absolute_error(df['neighbors_taux_de_couverture'], np.full(len(df['neighbors_taux_de_couverture']),np.mean(df['neighbors_taux_de_couverture'])))
dicty={}
all_results=pd.DataFrame(columns=['Value'], data=[[""]])

col_buttons = st.columns(4)
if col_buttons[0].button('Make the magic happen'):
    # Set params:
    if breakeven=='None':
        breakeven=None
    else:
        breakeven=float(breakeven)

    params=dict(
        select_meds=selection_medecins,
        sort_how=sortby,
        select_radius=radius,
        breakeven_rate=None,
        neighbors_weight=weight,
        min_neighbors=nb_neighbors,
        code_region=region_val,
    )

    req = requests.get(url, params=params)
    
    dicty = json.loads(req.json()['Data'])
    col_buttons[2].markdown(f'''### Evolution du taux: ''')
    if req.json()['Evolution du taux'] > 0:
        col_buttons[3].markdown(f'''### +{req.json()['Evolution du taux']*100:.2f} %''')
    else:
        col_buttons[3].markdown(f'''## Evolution du taux: {req.json()['Evolution du taux']*100:.2f} %''')
    res_dict = {
        'Initial rate':f"{req.json()['Ancien_taux_moyenne_communes']*100:.2f}%",
        'Calculated rate':f"{req.json()['Nouveau_taux_moyenne_communes']*100:.2f}%", 
        'Average moved distance':f"{req.json()['Distance_moyenne_parcourue']:.2f} km per GP", 
        'Total distance':f"{req.json()['Distance_totale_parcourue']} km", 
        'Number of relocated GPs':f"{int(float(req.json()['Distance_totale_parcourue'])/float(req.json()['Distance_moyenne_parcourue']))} GPs",
    }
    all_results=pd.DataFrame.from_dict(data=res_dict, orient='index', columns=['Values'] )
    mae_out=mean_absolute_error(df_combine['neighbors_taux_de_couverture'], np.full(len(df_combine['neighbors_taux_de_couverture']),np.mean(df_combine['neighbors_taux_de_couverture'])))
    #col_buttons[4].markdown(f'''{(mae_out-mae_init)*100:.2f}''')
    df_from_dicty = pd.DataFrame(dicty).reset_index().rename(columns={'index': 'code_insee'})
    df_combine=df_combine.rename(columns={'neighbors_taux_de_couverture':'neighbors_old_taux_de_couverture'}).merge(df_from_dicty, how='left', left_on='code_insee', right_on='code_insee')

if col_buttons[1].button('Return to original state'):
    df_combine.rename(columns={'neighbors_taux_de_couverture': 'calculated_neighbors_taux_de_couverture','neighbors_old_taux_de_couverture': 'neighbors_taux_de_couverture'})

lowest = df_combine[df_combine['neighbors_taux_de_couverture']<=0.6]
low_mid = df_combine[(df_combine['neighbors_taux_de_couverture']>0.6)&(df_combine['neighbors_taux_de_couverture']<=0.8)]
mid = df_combine[(df_combine['neighbors_taux_de_couverture']>0.8)&(df_combine['neighbors_taux_de_couverture']<=1.0)]
high_mid = df_combine[(df_combine['neighbors_taux_de_couverture']>1.0)&(df_combine['neighbors_taux_de_couverture']<=1.2)]
highest = df_combine[df_combine['neighbors_taux_de_couverture']>1.2]

def heatmap():
    filepath = "brouillon/df_api_test.csv"#https://raw.githubusercontent.com/giswqs/leafmap/master/examples/data/us_cities.csv"
    m = leafmap.Map(tiles="openstreetmap", center=(df_combine['Lat_commune'].mean(), df_combine['Lon_commune'].mean()), draw_export=True, zoom=8)
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

output_col = st.columns(2)
res_df_slot = output_col[0].empty()
results_dataframe = res_df_slot.dataframe(all_results)

# Statistiques
def stat_retour(df_avant, df_apres, name_col_av, name_col_ap):
    df_stats = df_avant.rename(columns={name_col_av: "taux_avant"})\
                            .merge(df_apres.rename(columns={name_col_ap: "taux_apres"}), on="code_insee")[["code_insee", "taux_avant","taux_apres"]]\
                            #.merge(region_df_name(), on="code_regions")

    plt.hist([df_stats["taux_avant"], df_stats["taux_apres"]], bins=[0,0.5,1,1.5, 2,3], color=["tab:orange", "tab:blue"])
    plt.title("Number of communes by coverage after transformation")
    plt.legend(["Before", "After"])
    plt.show()

fig, ax = plt.subplots()
ax = stat_retour(df,df_combine,'neighbors_taux_de_couverture', 'neighbors_taux_de_couverture')
output_col[1].pyplot(fig)

