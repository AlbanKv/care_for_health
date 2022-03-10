import pandas as pd
import matplotlib.pyplot as plt
from care_for_health import get_data
import seaborn as sns

############ STAT 1 ##############

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

    fig = plt.figure(1, figsize=(12, 7))

    bars = plt.barh(regions, taux_couv, color=colors)

    plt.xlim(0.6, 1.01)
    plt.title("Taux de couverture medecins de la France et par région")

    for bar in bars:
            width = bar.get_width()
            label_y = bar.get_y() + bar.get_height() / 2
            plt.text(width, label_y, s=f'{width}', color='black')

    return bars


############ STAT 2 ##############

def df_stat_regions_2():
    region_dict = {84:"Auvergne-Rhône-Alpes", 27:"Bourgogne-Franche-Comté",
                   53:"Bretagne", 24:"Centre-Val de Loire", 94:"Corse",
                   44:"Grand Est", 32:"Hauts-de-France", 11:"Île-de-France",
                   28:"Normandie", 75:"Nouvelle-Aquitaine", 76:"Occitanie",
                   52:"Pays de la Loire", 93:"Provence-Alpes-Côte d'Azur"}

    region_df=pd.DataFrame(region_dict.items(), columns=['code_regions','regions'])
    region_df = region_df.sort_values(by = 'code_regions').reset_index(drop=True)

    data_regions= get_data.get_data_region().copy()

    data_regions = data_regions.merge(region_df[['code_regions', 'regions']], how = 'left')

    first_col = data_regions.pop('regions')
    data_regions.insert(1, 'regions', first_col)

    data_regions['med_pour_100000_hab']=round(data_regions['Medecin_generaliste']/data_regions['Population_2018']*100000,1)

    data_regions['besoin_pour_100000_hab']=round(data_regions['Besoin_medecins']/data_regions['Population_2018']*100000,1)


    graph_data = data_regions[['code_regions','regions','med_pour_100000_hab','besoin_pour_100000_hab']]

    fr_graph_data = pd.DataFrame([[100,
                                   'France',
                                   round(data_regions['Medecin_generaliste'].sum()/data_regions['Population_2018'].sum()*100000,1),
                                   round(data_regions['Besoin_medecins'].sum()/data_regions['Population_2018'].sum()*100000,1)]],
                                 columns=['code_regions','regions','med_pour_100000_hab','besoin_pour_100000_hab'])

    concat_df = (graph_data, fr_graph_data)

    graph_data_full = pd.concat(concat_df)

    graph_data_full['diff']=graph_data_full['besoin_pour_100000_hab']-graph_data_full['med_pour_100000_hab']

    return graph_data_full

def stat_2():
    graph_data_full=df_stat_regions_2()
    df_sort = graph_data_full.sort_values(by = 'diff').reset_index(drop=True)

    barWidth = 0.4
    labels = df_sort['regions'].tolist()
    med_pour_100000_hab = df_sort['med_pour_100000_hab'].tolist()
    besoin_pour_100000_hab = df_sort['besoin_pour_100000_hab'].tolist()
    r1 = range(len(med_pour_100000_hab))
    r2 = [x + barWidth for x in r1]


    fig, ax = plt.subplots(figsize=(15, 7.3))
    rect1 = ax.bar(r1, med_pour_100000_hab, width = barWidth, label='Nb_medecins_actuels', color = ['lightgreen' for i in med_pour_100000_hab], linewidth = 2)
    rect2 = ax.bar(r2, besoin_pour_100000_hab, width = barWidth, label='Besoin_en_medecins', color = ['forestgreen' for i in besoin_pour_100000_hab], linewidth = 4)

    for bar in ax.patches:
        value = bar.get_height()
        text = f'{value}'
        text_x = bar.get_x() + bar.get_width() / 2
        text_y = bar.get_y() + value
        ax.text(text_x, text_y, text, ha='center',color='black',size=12)

    # Add some text for labels,x-axis tick labels
    ax.set_ylabel('pour 100 000 habitants')
    ax.set_xlabel('Régions')
    ax.set_ylim(55, 100)
    ax.set_xticks(r1)
    ax.set_xticklabels(labels,rotation=90)
    ax.legend(loc='best')

    plt.title("Nombre de Medecins actuels et besoin en medecins pour 100 000 habitants : France et région \n(trié par la différence entre besoin et actuel)")

    return plt


############ STAT 3 ##############

def stat_3(data_com_neig):
    data_com_fr = get_data.get_full_medbase(region=None)
    df_1 = data_com_fr[["code_regions", "code_insee"]].groupby("code_regions", as_index=False).count().rename(columns={"code_insee": "count_communes"})
    tx_couv = tx_couv_regions_fr()
    tx_couv.rename(columns={"taux_de_couverture": "tx_couv_global_region"}, inplace=True)

    data_com_neig_1= data_com_neig.copy()
    data_com_neig_1 = data_com_neig_1.merge(tx_couv, on='code_regions', how='left')

    df_exc = data_com_neig_1[data_com_neig_1["neighbors_taux_de_couverture"] > (data_com_neig_1['tx_couv_global_region']*1.1)].groupby("code_regions", as_index=False)\
                                                    .count()[["code_regions", "code_insee"]]\
                                                    .rename(columns={"code_insee": "count_exc"})\
                                                    .sort_values("count_exc")
    df_exc = df_exc.join(region_df_name().set_index("code_regions")["regions"], on="code_regions", how="left").sort_values("count_exc", ascending=True)

    df_1 = data_com_fr[["code_regions", "code_insee"]].groupby("code_regions", as_index=False).count().rename(columns={"code_insee": "count_communes"})

    df_exc = df_exc.merge(df_1, on="code_regions")
    df_exc["per_cent_exc"] = round(df_exc["count_exc"] / df_exc["count_communes"] * 100, 0).astype(int)

    fr_graph_df_1 = pd.DataFrame([[100,
                                 df_exc["count_exc"].sum(),
                                 'France',
                                 df_exc["count_communes"].sum(),
                                 round(df_exc["count_exc"].sum() / df_exc["count_communes"].sum() * 100, 0).astype(int),
                                ]],
                               columns=['code_regions','count_exc','regions','count_communes','per_cent_exc'])

    concat_df_1 = (df_exc, fr_graph_df_1)

    df_exc = pd.concat(concat_df_1)

    df_def = data_com_neig_1[data_com_neig_1["neighbors_taux_de_couverture"] > (data_com_neig_1['tx_couv_global_region']*0.9)].groupby("code_regions", as_index=False)\
                                                    .count()[["code_regions", "code_insee"]]\
                                                    .rename(columns={"code_insee": "count_def"})\
                                                    .sort_values("count_def")
    df_def = df_def.join(region_df_name().set_index("code_regions")["regions"], on="code_regions", how="left").sort_values("count_def", ascending=True)

    df_1 = data_com_fr[["code_regions", "code_insee"]].groupby("code_regions", as_index=False).count().rename(columns={"code_insee": "count_communes"})

    df_def = df_def.merge(df_1, on="code_regions")
    df_def["per_cent_def"] = round(df_def["count_def"] / df_def["count_communes"] * 100, 0).astype(int)

    fr_graph_df = pd.DataFrame([[100,
                                 df_def["count_def"].sum(),
                                 'France',
                                 df_def["count_communes"].sum(),
                                 round(df_def["count_def"].sum() / df_def["count_communes"].sum() * 100, 0).astype(int),
                                ]],
                               columns=['code_regions','count_def','regions','count_communes','per_cent_def'])

    concat_df = (df_def, fr_graph_df)

    df_def = pd.concat(concat_df)


    df_def=df_def[['code_regions','regions','per_cent_def']]
    df_exc=df_exc[['code_regions','per_cent_exc']]

    df_exc_def = df_def.merge(df_exc, on="code_regions")

    df_exc_def['diff']=df_exc_def['per_cent_def']-df_exc_def['per_cent_exc']

    df_sort = df_exc_def.sort_values(by = 'diff').reset_index(drop=True)

    barWidth = 0.4
    labels = df_sort['regions'].tolist()
    per_cent_exc = df_sort['per_cent_exc'].tolist()
    per_cent_def = df_sort['per_cent_def'].tolist()
    r1 = range(len(per_cent_exc))
    r2 = [x + barWidth for x in r1]


    fig, ax = plt.subplots(figsize=(15, 10))
    rect1 = ax.bar(r1, per_cent_exc, width = barWidth, label='Communes ou le taux de couverture est 10% plus élevé que le taux de couverture régionnal (%)', color = ['lightgreen' for i in per_cent_exc], linewidth = 2)
    rect2 = ax.bar(r2, per_cent_def, width = barWidth, label='Communes ou le taux de couverture est 10% moins élevé que le taux de couverture régionnal (%)', color = ['forestgreen' for i in per_cent_def], linewidth = 4)

    for bar in ax.patches:
        value = bar.get_height()
        text = f'{value}'
        text_x = bar.get_x() + bar.get_width() / 2
        text_y = bar.get_y() + value
        ax.text(text_x, text_y, text, ha='center',color='black',size=12)

    # Add some text for labels,x-axis tick labels
    ax.set_ylabel('en pourcentage (%)')
    ax.set_xlabel('Régions')
    ax.set_xticks(r1)
    ax.set_xticklabels(labels,rotation=90)
    ax.legend(loc='best')

    plt.title("Pourcentage de communes au dessous et en dessous du taux de couverture régionnal: France et région \n(trié par la différence entre les 2 pourcentages % de communes)")

    return plt


############ STAT 4 ##############

def coverage_rate_range(df,col_cov_rate_str):
    fig_2, ax_2 = plt.subplots(figsize=(15,7))
    sns.histplot(df[col_cov_rate_str], bins=[0,0.2,0.4,0.6, 0.8, 1.0 ,1.2,1.4, 1.6, 1.8, 2.0], stat="percent");
    return fig_2
