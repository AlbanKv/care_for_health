import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.markdown('''
# General physicists repartition
## 
''')

columns = st.columns(4)

pickup_longitude = columns[0].text_input('Pickup longitude', value='40.7614327')
pickup_latitude = columns[1].text_input('Pickup latitude', value='-73.9798156')
dropoff_longitude = columns[2].text_input('Dropoff longitude', value='40.6513111')
dropoff_latitude = columns[3].text_input('Dropoff latitude', value='-73.8803331')

st.markdown('''
### Please enter your travel details:
''')

columns2 = st.columns(3)
pickup_date = columns2[0].date_input('Date')#, value='2012-10-06 2012:10:20')
pickup_time = columns2[1].time_input('Time')#, value='2012-10-06 2012:10:20')
passenger_count = columns2[2].selectbox('N. of passengers', range(1,11))
location_df = pd.DataFrame([[float(pickup_longitude), float(pickup_latitude)]], columns=['lat', 'lon'])

url = 'https://fr-nantes-acy-img1-gfiqg24vta-ew.a.run.app/predict'

pickup_datetime = f'{pickup_date} {pickup_time}'
params = dict(
  pickup_datetime=pickup_datetime,#'2012-10-06 12:10:20',
  pickup_longitude=pickup_longitude,
  pickup_latitude=pickup_latitude,
  dropoff_longitude=dropoff_longitude,
  dropoff_latitude=dropoff_latitude,    
  passenger_count=passenger_count
)

if st.button('Calculate my fare'):
    # print is visible in the server output, not in the page
    print('button clicked!')
    req = requests.get(url, params=params)
    st.markdown(f'''
        ## {req.json()['fare']:.2f}
        ''')
else:
    st.write('Click me!')


# @st.cache
def plot_map():
    return px.line_mapbox(my_trip, lat='lat', lon='lon', zoom=3, height=400)
    
px.scatter_mapbox(my_trip, lat='lat', lon='lon')

fig = plot_map()

fig.update_layout(mapbox_style="stamen-terrain", mapbox_zoom=10, mapbox_center_lat = mid_lat, mapbox_center_lon = mid_lon,
    margin={"r":0,"t":0,"l":0,"b":0})

st.plotly_chart(fig, use_container_width=True)



fig.update_layout()

#fig = chloropleth_map_communes(df_communes, code_insee_str, taux_couv_str)

## MAP PAR COMMUNES (SELECTION DES REGIONS A FAIRE DANS LE DATAFRAME)
def chloropleth_map_communes(df_communes,code_insee_str,taux_couv_str):
    """
    arguments :
        code_insee_str: mettre le nom de la colonne ou le code INSEE est présent ('code_insee')
        taux_couv_str: mettre le nom de la colonne ou le taux de couverture est présent ('taux_de_couverture')
    """
    #json_load
    json_data = 'communes_fr.json'
    json_load = json.load(open(json_data))

    #map
    fig = go.Figure(go.Choroplethmapbox(
            geojson = json_load, #Assign geojson file : délimitation des régions
            featureidkey = "properties.codgeo", #Assign feature key : code INSEE
            locations = df_communes[code_insee_str], #Assign location data : code INSEE
            z = df_communes[taux_couv_str], #Assign information data : taux de couverture
            zmin=0, zmax=1.5,
            colorscale = [[0, 'rgb(0,0,255)'], [0.3, 'rgb(0,255,0)'], [1, 'rgb(255,0,0)']],
            showscale = True))

    fig.update_layout(
        width = 600,
        height = 600,
        mapbox_style = "carto-positron",
        mapbox_zoom = 4,
        mapbox_center = {"lat": 46.227638, "lon": 2.213749}, #Centre de la France
    )

    return fig



# st.write(req.get('fare'))
