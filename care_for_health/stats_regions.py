import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from care_for_health import get_data
import seaborn as sns

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

############ STAT 1 ##############

def bar_plot_regions_fr():
    df_sort = pd.read_csv("../raw_data/fr_reg_stats_data_light.csv", delimiter=',', encoding='utf-8')

    df_sort = df_sort.sort_values(by = 'taux_de_couverture').reset_index(drop=True)

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
    plt.title("Coverage rate of general practitioners in France and by region")

    for bar in bars:
            width = bar.get_width()
            label_y = bar.get_y() + bar.get_height() / 2
            plt.text(width, label_y, s=f'{width}', color='black')

    return bars

############ STAT 2 ##############

def actual_need_gp():
    graph_data_full=pd.read_csv("../raw_data/fr_reg_stats_data_light.csv", delimiter=',', encoding='utf-8')

    graph_data_full['diff']=graph_data_full['besoin_pour_100000_hab']-graph_data_full['med_pour_100000_hab']

    df_sort = graph_data_full.sort_values(by = 'diff').reset_index(drop=True)

    barWidth = 0.4
    labels = df_sort['regions'].tolist()
    med_pour_100000_hab = df_sort['med_pour_100000_hab'].tolist()
    besoin_pour_100000_hab = df_sort['besoin_pour_100000_hab'].tolist()
    r1 = range(len(med_pour_100000_hab))
    r2 = [x + barWidth for x in r1]


    fig, ax = plt.subplots(figsize=(15, 7.3))
    rect1 = ax.bar(r1, med_pour_100000_hab, width = barWidth, label='Actual_nb_GP', color = ['lightgreen' for i in med_pour_100000_hab], linewidth = 2)
    rect2 = ax.bar(r2, besoin_pour_100000_hab, width = barWidth, label='Need_nb_GP', color = ['forestgreen' for i in besoin_pour_100000_hab], linewidth = 4)

    for bar in ax.patches:
        value = bar.get_height()
        text = f'{value}'
        text_x = bar.get_x() + bar.get_width() / 2
        text_y = bar.get_y() + value
        ax.text(text_x, text_y, text, ha='center',color='black',size=12)

    # Add some text for labels,x-axis tick labels
    ax.set_ylabel('per 100 000 persons')
    ax.set_xlabel('regions')
    ax.set_ylim(55, 100)
    ax.set_xticks(r1)
    ax.set_xticklabels(labels,rotation=90)
    ax.legend(loc='best')

    plt.title("Number of current GP and need for GP per 100 000 persons: France and region \n(sorted by the difference between need and actual)")

    return plt


############ STAT 3 ##############

def municipalities_exc_def():
    category_names = ['% surplus municipalities',"% municipalities +/- 10% of region's coverage rate", '% deficit municipalities']

    df= pd.read_csv("../raw_data/fr_reg_stats_data_light.csv", delimiter=',', encoding='utf-8')
    df = df.sort_values(by = 'per_cent_exc').reset_index(drop=True)

    results = {
        df["regions"][0]: [df["per_cent_exc"][0], df["per_cent_mean"][0], df["per_cent_def"][0]],
        df["regions"][1]: [df["per_cent_exc"][1], df["per_cent_mean"][1], df["per_cent_def"][1]],
        df["regions"][2]: [df["per_cent_exc"][2], df["per_cent_mean"][2], df["per_cent_def"][2]],
        df["regions"][3]: [df["per_cent_exc"][3], df["per_cent_mean"][3], df["per_cent_def"][3]],
        df["regions"][4]: [df["per_cent_exc"][4], df["per_cent_mean"][4], df["per_cent_def"][4]],
        df["regions"][5]: [df["per_cent_exc"][5], df["per_cent_mean"][5], df["per_cent_def"][5]],
        df["regions"][6]: [df["per_cent_exc"][6], df["per_cent_mean"][6], df["per_cent_def"][6]],
        df["regions"][7]: [df["per_cent_exc"][7], df["per_cent_mean"][7], df["per_cent_def"][7]],
        df["regions"][8]: [df["per_cent_exc"][8], df["per_cent_mean"][8], df["per_cent_def"][8]],
        df["regions"][9]: [df["per_cent_exc"][9], df["per_cent_mean"][9], df["per_cent_def"][9]],
        df["regions"][10]: [df["per_cent_exc"][10], df["per_cent_mean"][10], df["per_cent_def"][10]],
        df["regions"][11]: [df["per_cent_exc"][11], df["per_cent_mean"][11], df["per_cent_def"][11]],
        df["regions"][12]: [df["per_cent_exc"][12], df["per_cent_mean"][12], df["per_cent_def"][12]],
        df["regions"][13]: [df["per_cent_exc"][13], df["per_cent_mean"][13], df["per_cent_def"][13]]
    }
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('rainbow')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.1 else 'black'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(int(c)), ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax


############ STAT 4 ##############

def coverage_rate_range():
    df_1 = pd.read_csv("../raw_data/data_coverage_rate_range.csv", delimiter=',', encoding='utf-8')
    fig_2, ax_2 = plt.subplots(figsize=(15,7))
    ax_2.set_xlabel('neighbors coverage rate')
    sns.histplot(df_1['neighbors_taux_de_couverture'], bins=[0,0.2,0.4,0.6, 0.8, 1.0 ,1.2,1.4, 1.6, 1.8, 2.0], stat="percent");
    return fig_2










'''
Deprecated function
'''

"""############ STAT 1 ##############

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
    plt.title("Coverage rate of general practitioners in France and by region")

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
    rect1 = ax.bar(r1, med_pour_100000_hab, width = barWidth, label='Actual_nb_GP', color = ['lightgreen' for i in med_pour_100000_hab], linewidth = 2)
    rect2 = ax.bar(r2, besoin_pour_100000_hab, width = barWidth, label='Need_nb_GP', color = ['forestgreen' for i in besoin_pour_100000_hab], linewidth = 4)

    for bar in ax.patches:
        value = bar.get_height()
        text = f'{value}'
        text_x = bar.get_x() + bar.get_width() / 2
        text_y = bar.get_y() + value
        ax.text(text_x, text_y, text, ha='center',color='black',size=12)

    # Add some text for labels,x-axis tick labels
    ax.set_ylabel('per 100 000 persons')
    ax.set_xlabel('regions')
    ax.set_ylim(55, 100)
    ax.set_xticks(r1)
    ax.set_xticklabels(labels,rotation=90)
    ax.legend(loc='best')

    plt.title("Number of current GP and need for GP per 100 000 persons: France and region \n(sorted by the difference between need and actual)")

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

############ STAT 3.2 ##############

def stat_3_good():
    category_names = ['% surplus municipalities',"% municipalities +/- 10% of region's coverage rate", '% deficit municipalities']

    results = {
        'Île-de-France': [2, 100-2-56, 56],
        'Pays de la Loire': [14, 100-14-36, 36],
        'Occitanie': [14, 100-14-39, 39],
        'Hauts-de-France': [14, 100-14-41, 41],
        'Nouvelle-Aquitaine': [15, 100-15-38, 38],
        'Centre-Val de Loire': [17, 100-17-45, 45],
        'France': [17, 100-17-47, 47],
        'Normandie': [17, 100-17-49, 49],
        'Bretagne': [17, 100-17-55, 55],
        'Auvergne-Rhône-Alpes': [19, 100-19-52, 52],
        'Grand Est': [20, 100-20-51, 51],
        "Provence-Alpes-Côte d'Azur": [21, 100-21-51, 51],
        'Corse': [22, 100-22-65, 65],
        'Bourgogne-Franche-Comté': [28, 100-28-57, 57]
    }
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('rainbow')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.1 else 'black'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(int(c)), ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax

############ STAT 4.1 ##############

def coverage_rate_range(df,col_cov_rate_str,title_x):
    fig_2, ax_2 = plt.subplots(figsize=(15,7))
    #"neighbors coverage rate"
    ax_2.set_xlabel(title_x)
    sns.histplot(df[col_cov_rate_str], bins=[0,0.2,0.4,0.6, 0.8, 1.0 ,1.2,1.4, 1.6, 1.8, 2.0], stat="percent");
    return fig_2
"""
