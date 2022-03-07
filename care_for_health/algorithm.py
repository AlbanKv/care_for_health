from haversine import haversine, Unit 
from sklearn.neighbors import NearestNeighbors
from care_for_health import preprocessing
import numpy as np
import pandas as pd
import random

### ---------------- Algorithme: ---------------------------------------
# Cherche à redistribuer de proche en proche les médecins en excédent 
# Recalcule le taux de couverture
# Devra, à terme, mémoriser les modifications apportées

def algorithm_all_available_med(df, selection_medecins='tous', sortby='rate', radius=1, moy_region=0.84, recalcul=True):
    '''
    Pour un dataframe donné (df), redispatch les médecins en fonction de plusieurs paramètres : 
        * selection_medecins: 'tous', 'excédent'. 
            'tous' signifie que tous les médecins présents dans la commune seront redispatchés,
            'excédent' signifie que seuls les médecins en plus dans la commune et le voisinage seront redispatchés.
        * sortby: 'distance', 'rate', 'max_number', 'random'.
            'distance' signifie que le plus proche reçoit le 1er des médecins
            'rate' signifie que la commune avec le taux le plus faible reçoit en premier des médecins
            'max_number' signifie que la commune avec le plus gros déficit en absolu reçoit en premier des médecins
            'random', random
            Renvoie l'index de chaque voisin, avec 3 informations : 
            > Distance, en °
            > Le taux de couverture (avec voisinage)
            > Le déficit en médecins (avec voisinage) (valeur positive = manque de médecin, valeur négative = surplus de médecin)
        * radius: float. 
            Rayon autour duquel les médecins pourront se déplacer. 1 ~ 14km
        * moy_region: float.
            Score au-dessus duquel une commune est identifiée comme excédentaire
        * recalcul: True / False
            A l'issue de chaque traitement d'une commune, un recalcul des taux_de_couverture est réalisé si = True
    A développer : réaliser un dispatch non pas linéaire (chaque commune récupère 1 médecin jusqu'à épuisement du stock)
    mais en fonction du nombre de médecins manquants.
    '''
    df_ = df.copy()

    #Initialisation des plus proches voisins : 
    rnc = NearestNeighbors(radius=radius, p=2)
    rnc.fit(df_[['Lat_commune', 'Lon_commune']])
    itr=0

    # Itération sur les codes INSEE : 
    pool_communes = df_[(df_.neighbors_taux_de_couverture > moy_region)&(df_.Medecin_generaliste >= 1)]

    # Baseline avant lancement de l'itération :
    baseline = df_.neighbors_taux_de_couverture.mean()
    baseline_communes = len(df[df['neighbors_taux_de_couverture']<moy_region])

    # Listing des modifications :
    distance = []
    output = []
    recap = {}
    trades = []
    neighbors_stats = []

    # Début de l'itération
    for ind, row in pool_communes.iterrows():

        # Nombre de médecins disponibles AVANT transformation:
        med_dispo = nb_medecins_disponibles(df.loc[ind,:], selection_medecins=selection_medecins)

        # Identification des communes ayant une meilleure couverture que la moyenne de la région:
        if df_.loc[ind,'neighbors_taux_de_couverture'] > moy_region and med_dispo >= 1 :
            
            # Enregistrement des informations des nearest neighbors et cleaning avec seulement les déficitaires:
            # Idée pour la suite: ne pourrait-on pas tester chaque type de 'sortby' et conserver le meilleur résultat ?
            neighbors_dict = sort_neighbors(ind, rnc, df_, sortby=sortby, moy_region=moy_region, radius=radius)

            # Opération sur les nearest neighbors:
            transfer = False
            count = len(list(neighbors_dict.keys()))
            if count<= 0:
                pass
            else:
                while transfer==False: 
                    ### Approche dispatch d'un seul médecin:
                    for key in list(neighbors_dict.keys()):
                        count -= 1
                        prev = df_.loc[key, 'Medecin_generaliste']
                        print(neighbors_dict[key])
                        df_.loc[key, 'Medecin_generaliste'] = df_.loc[key, 'Medecin_generaliste'] + 1
                        df_.loc[ind, 'Medecin_generaliste'] = df_.loc[ind, 'Medecin_generaliste'] - 1
                        med_dispo -= 1
                        trades.append([(df_.loc[ind,'code_insee'], df_.loc[key,'code_insee'], prev, df_.loc[key, 'Medecin_generaliste'])])
                        distance.append(haversine((df_.loc[key, 'Lat_commune'],df_.loc[key, 'Lon_commune']),(df.loc[ind, 'Lat_commune'],df.loc[ind, 'Lon_commune'])))
                        if med_dispo <= 0:
                            transfer = True
                if recalcul==True:
                    itr+=1
                    df_ = preprocessing.get_meds_neighbors_df(df_)
                    if itr%10==0:
                        print(f'Traitement n°{itr}')

    # Production du dictionnaire récapitulatif & chiffres clés
    df_ = preprocessing.get_meds_neighbors_df(df_)
    output = df_.neighbors_taux_de_couverture.mean()
    output_communes = len(df_[df_['neighbors_taux_de_couverture']<moy_region])

    recap['trades'] = trades
    recap['distances'] = distance
    recap['scores'] = {
        'ancien_taux':baseline,
        'nouveau_taux':output,
        'delta_taux': output-baseline,
        'ancien_communes_déficitaires': baseline_communes,
        'nouveau_communes_déficitaires': output_communes,
        'delta_communes': output_communes-baseline_communes,
        }
    recap['nb_neighbors'] = neighbors_stats

    print(f"Nb de communes en déficit avant : {baseline_communes} - après : {output_communes}\n")
    print(f"Distance des médecins déplacés:\nMoyenne: {np.array(distance).mean():.2f}\nMin: {np.array(distance).min():.2f} - Max: {np.array(distance).max():.2f}")
    print(f"Médecins déplacés : {len(distance)}\n")
    print(f"Gain en points de: {(output - baseline)*100:.3f}%\navant : {baseline*100:.2f}% - après : {output.max()*100:.2f}%")
    return df_, recap


def nb_medecins_disponibles(row, selection_medecins='excédent'):
    if selection_medecins == 'excédent':
        if row.neighbors_nb_medecins-row.neighbors_Besoin_medecins < 0:
            return 0
        return min(int(row.Medecin_generaliste), int(row.neighbors_nb_medecins-row.neighbors_Besoin_medecins))
    elif selection_medecins == 'tous':
        return int(row.Medecin_generaliste)
    else:
        return 0

def sort_neighbors(ind, rnc, df_, sortby='distance', moy_region=0.84, radius=1):
    '''
    Renvoie l'index de chaque voisin, avec 3 informations : 
        * Distance, en °
        * Le taux de couverture (avec voisinage)
        * Le déficit en médecins (avec voisinage) (valeur positive = manque de médecin, valeur négative = surplus de médecin)
    sortby peut prendre les valeurs suivantes : 'distance', 'rate', 'max_number', 'random'
    '''
    closest = rnc.radius_neighbors(X=[[df_.loc[ind,'Lat_commune'],df_.loc[ind,'Lon_commune']]],radius=radius/80*8)
    neighbors_list = list(closest[1][0])
    list_ind = neighbors_list.index(ind)
    neighbors_list.pop(list_ind)
    neighbors_dist = list(closest[0][0])
    neighbors_dist.pop(list_ind)
    neighbors_rate=np.array(df_.loc[(df_.index.isin(neighbors_list))]['neighbors_taux_de_couverture'])
    neighbors_code_insee=np.array(df_.loc[(df_.index.isin(neighbors_list))]['code_insee'])
    neighbors_deficit_local=np.array(df_.loc[(df_.index.isin(neighbors_list))]['Medecin_generaliste'])-np.array(df_.loc[(df_.index.isin(neighbors_list))]['Besoin_medecins'])
    neighbors_deficit_neighbors=np.array(df_.loc[(df_.index.isin(neighbors_list))]['neighbors_nb_medecins'])-np.array(df_.loc[(df_.index.isin(neighbors_list))]['neighbors_Besoin_medecins'])
    neighbors_intermediate_list=list(zip(neighbors_dist, neighbors_rate, neighbors_deficit_neighbors.astype('int'), neighbors_deficit_local.astype('int'), neighbors_code_insee))
    neighbors_dict=dict(zip(neighbors_list, neighbors_intermediate_list))
    if sortby=='distance':
        neighbors_dict={k: v for k, v in sorted(neighbors_dict.items(), key=lambda item: item[1][0])}
    elif sortby=='rate':
        neighbors_dict={k: v for k, v in sorted(neighbors_dict.items(), key=lambda item: item[1][1])}
    elif sortby=='max_number':
        neighbors_dict={k: v for k, v in sorted(neighbors_dict.items(), key=lambda item: item[1][2])}
    else:
        key_list = list(neighbors_dict)
        random.shuffle(key_list)
        n2 = {}
        for key in key_list:
            n2[key]=neighbors_dict[key]
        neighbors_dict = n2.copy()

    for key in list(neighbors_dict.keys()):
        if neighbors_dict[key][1] > moy_region:
            del neighbors_dict[key]

    return neighbors_dict

def algorithm_1_med(df, param='excédent'):
    df_ = df.copy()

    moy_region = 0.81

    #Initialisation des plus proches voisins : 
    rnc = NearestNeighbors(radius=10, p=2)
    rnc.fit(df_[['Lat_commune', 'Lon_commune']])

    # Itération sur les codes INSEE : 
    pool_communes = df_[(df_.neighbors_taux_de_couverture > moy_region)&(df_.Medecin_generaliste >= 1)]

    # Baseline avant lancement de l'itération :
    baseline = df_.neighbors_taux_de_couverture.mean()
    baseline_communes = len(df[df['neighbors_taux_de_couverture']<moy_region])

    # Listing des modifications :
    distance = []
    output = []
    recap = {}
    trades = []
    neighbors_stats = []

    # Début de l'itération
    for ind, row in pool_communes.iterrows():

        # Nombre de médecins disponibles :
        med_dispo = nb_medecins_disponibles(df_.loc[ind,:], param=param)

        # Identification des communes ayant une meilleure couverture que la moyenne de la région:
        if df_.loc[ind,'neighbors_taux_de_couverture'] > moy_region and med_dispo >=1 :
            
            # Enregistrement des informations des nearest neighbors:
            closest = rnc.radius_neighbors(X=[[df_.loc[ind,'Lat_commune'],df_.loc[ind,'Lon_commune']]],radius=1/80*8)
            neighbors_list = list(closest[1][0])
            list_ind = neighbors_list.index(ind)
            neighbors_list.pop(list_ind)
            neighbors_dist = list(closest[0][0])
            neighbors_dist.pop(list_ind)
            neighbors_dict=dict(zip(neighbors_list, neighbors_dist))
            neighbors_dict={k: v for k, v in sorted(neighbors_dict.items(), key=lambda item: item[1])}
            neighbors_stats.append(len(neighbors_dict))

            # Opération sur les nearest neighbors:
            transfer = False
            while transfer==False: 
                count = len(list(neighbors_dict.keys()))
                for key in list(neighbors_dict.keys()):
                    count -= 1
                    if df_.loc[key, 'neighbors_taux_de_couverture'] < moy_region:
                        prev = df_.loc[key, 'Medecin_generaliste']
                        df_.loc[key, 'Medecin_generaliste'] = df_.loc[key, 'Medecin_generaliste'] + 1
                        df_.loc[ind, 'Medecin_generaliste'] = df_.loc[ind, 'Medecin_generaliste'] - 1
                        trades.append([(df_.loc[ind,'code_insee'], df_.loc[key,'code_insee'], prev, df_.loc[key, 'Medecin_generaliste'])])
                        distance.append(haversine((df_.loc[key, 'Lat_commune'],df_.loc[key, 'Lon_commune']),(df.loc[ind, 'Lat_commune'],df.loc[ind, 'Lon_commune'])))
                        transfer = True
                    if count <= 0:
                        transfer = True

    df_ = preprocessing.get_meds_neighbors_df(df_)
    output = df_.neighbors_taux_de_couverture.mean()
    output_communes = len(df_[df_['neighbors_taux_de_couverture']<moy_region])

    recap['trades'] = trades
    recap['distances'] = distance
    recap['scores'] = {
        'ancien_taux':baseline,
        'nouveau_taux':output,
        'delta_taux': output-baseline,
        'ancien_communes_déficitaires': baseline_communes,
        'nouveau_communes_déficitaires': output_communes,
        'delta_communes': output_communes-baseline_communes,
        }
    recap['nb_neighbors'] = neighbors_stats

    print(f"Nb de communes en déficit avant : {baseline_communes} - après : {output_communes}\n")
    print(f"Distance des médecins déplacés:\nMoyenne: {np.array(distance).mean():.2f}\nMin: {np.array(distance).min():.2f} - Max: {np.array(distance).max():.2f}")
    print(f"Médecins déplacés : {len(distance)}\n")
    print(f"Gain en points de: {(output - baseline)*100:.3f}%\navant : {baseline*100:.2f}% - après : {output.max()*100:.2f}%")
    return df_, recap
