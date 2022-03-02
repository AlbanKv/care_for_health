import plotly.express as px
from plotly import graph_objects as go
import json

def col_delta(df):
    df['Delta_besoins_med_ge']=df['Besoin_medecins']-df['Medecin_generaliste']
    return df

#fonction Density Map interactive
def density_maps(df,lat,lon,delta):
    df = col_delta(df)
    fig_1 = px.density_mapbox(df, lat=lat, lon=lon, z=delta, radius=10,
                            center=dict(lat=47.351608, lon=-0.763770), zoom=6,
                            mapbox_style="stamen-terrain")
    return fig_1

#fonction Chloropleth Map interactive
def chloropleth_maps(df,col_code_insee,delta,json_data):
    df = col_delta(df)
    #json_load
    json_load = json.load(open(json_data)) #'communes_fr.json'

    #Create figure object
    fig_2 = go.Figure(go.Choroplethmapbox(
            geojson = json_load, #Assign geojson file
            featureidkey = "properties.codgeo", #Assign feature key
            locations = df[col_code_insee], #Assign location data
            z = df[delta], #Assign information data
            zauto = True,
            colorscale = 'viridis',
            showscale = True))

    #Update layout
    fig_2.update_layout(
        mapbox_style = "carto-positron", #Decide a style for the map
        mapbox_zoom = 6,
        mapbox_center = {"lat": 47.351608, "lon": -0.763770}, #Centre de la région PdL
    )

    return fig_2
