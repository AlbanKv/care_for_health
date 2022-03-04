from haversine import haversine, Unit 
from sklearn.neighbors import NearestNeighbors
from care_for_health import preprocessing
import numpy as np
import pandas as pd

### ---------------- Algorithme: ---------------------------------------
# Cherche à redistribuer de proche en proche les médecins en excédent 
# Recalcule le taux de couverture
# Devra, à terme, mémoriser les modifications apportées

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

def nb_medecins_disponibles(row, param='excédent'):
    if param == 'excédent':
        if row.neighbors_nb_medecins-row.neighbors_Besoin_medecins < 0:
            return 0
        return min(int(row.Medecin_generaliste), int(row.neighbors_nb_medecins-row.neighbors_Besoin_medecins))
    elif param == 'tous':
        return int(row.Medecin_generaliste)
    pass

