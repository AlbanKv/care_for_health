import pandas as pd
import matplotlib.pyplot as plt
from care_for_health import get_data

def region_df_name():
    region_dict = {84:"Auvergne-Rhône-Alpes",
                   27:"Bourgogne-Franche-Comté",
                   53:"Bretagne",
                   24:"Centre-Val de Loire",
                   94:"Corse",
                   44:"Grand Est",
                   32:"Hauts-de-France",
                   11:"Île-de-France",
                   28:"Normandie",
                   75:"Nouvelle-Aquitaine",
                   76:"Occitanie",
                   52:"Pays de la Loire",
                   93:"Provence-Alpes-Côte d'Azur",
                   100:"France"}

    region_df=pd.DataFrame(region_dict.items(), columns=['code_regions','regions'])
    return region_df

def tx_couv_regions_fr():
    regions_stat = get_data.get_data_region()
    region_df = region_df_name()

    value_list = regions_stat['taux_de_couverture'].astype(float).tolist()
    key_list = regions_stat['code_regions'].astype(int).tolist()

    dict_reg = dict(zip(key_list, value_list))
    dict_reg[100] = round(regions_stat['Medecin_generaliste'].sum()/regions_stat['Besoin_medecins'].sum(),2)
    df=pd.DataFrame(dict_reg.items(), columns=['code_regions','taux_de_couverture'])

    df = df.join(region_df.set_index('code_regions')['regions'], on='code_regions', how='left')

    first_column = df.pop('regions')
    df.insert(0, 'regions', first_column)
    df_sort = df.sort_values(by = 'taux_de_couverture').reset_index(drop=True)
    return df_sort

def bar_plot_regions_fr():
    df_sort = tx_couv_regions_fr()

    fr_mean =df_sort[df_sort['code_regions']==100]
    fr_mean = fr_mean['taux_de_couverture'].tolist()[0]

    regions = df_sort['regions'].tolist()
    taux_couv = df_sort['taux_de_couverture'].tolist()
    colors=[]
    for i in taux_couv:
        if i>fr_mean:
            colors.append('lightgreen')
        if i==fr_mean:
            colors.append('royalblue')
        if i<fr_mean:
            colors.append('lightcoral')

    fig = plt.figure(1, figsize=(10, 7))

    bars = plt.barh(regions, taux_couv, color=colors)

    plt.title("Taux de couverture medecins de la France et par région")

    for bar in bars:
            width = bar.get_width()
            label_y = bar.get_y() + bar.get_height() / 2
            plt.text(width, label_y, s=f'{width}', color='black')

    return bars
