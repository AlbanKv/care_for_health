import plotly.express as px
from plotly import graph_objects as go
import json

## MAP PAR REGIONS
def chloropleth_map_regions(df_regions,code_regions_str,taux_couv_str):
    """
    arguments :
        code_regions_str: mettre le nom de la colonne ou le code région est présent ('code_regions')
        taux_couv_str: mettre le nom de la colonne ou le taux de couverture est présent ('taux_de_couverture')
    """
    #json_load
    json_data = 'regions_fr.geojson'
    json_load = json.load(open(json_data))

    #map
    fig = go.Figure(go.Choroplethmapbox(
            geojson = json_load, #Assign geojson file : délimitation des régions
            featureidkey = "properties.code", #Assign feature key : code régions
            locations = df_regions[code_regions_str], #Assign location data : code régions
            z = df_regions[taux_couv_str], #Assign information data : taux de couverture
            zmin=df_regions[taux_couv_str].min(), zmax=df_regions[taux_couv_str].max(),
            colorscale = 'viridis',
            showscale = True))

    fig.update_layout(
        width = 600,
        height = 600,
        mapbox_style = "carto-positron",
        mapbox_zoom = 4,
        mapbox_center = {"lat": 46.227638, "lon": 2.213749}, #Centre de la France
    )

    return fig


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
            zmin=df_communes[taux_couv_str].min(), zmax=df_communes[taux_couv_str].max(),
            colorscale = 'viridis',
            showscale = True))

    fig.update_layout(
        width = 600,
        height = 600,
        mapbox_style = "carto-positron",
        mapbox_zoom = 4,
        mapbox_center = {"lat": 46.227638, "lon": 2.213749}, #Centre de la France
    )

    return fig
