from haversine import haversine, Unit 
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

from care_for_health import preprocessing
import numpy as np
import pandas as pd
import random

### ---------------- Algorithme: ---------------------------------------
# Cherche à redistribuer de proche en proche les médecins en excédent 
# Recalcule le taux de couverture
# Devra, à terme, mémoriser les modifications apportées

def algorithm_V2(df, selection_medecins='tous', sortby='calculated', radius=15, moy_region=0.84, recalcul=True, poids_des_voisins=0.1, nb_voisins_minimum=3):
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
    '''
    df_ = df.copy()

    #Initialisation des plus proches voisins : 
    rnc = NearestNeighbors(radius=radius*0.013276477888701137, p=2)
    rnc.fit(df_[['Lat_commune', 'Lon_commune']])
    itr=0

    # Itération sur les codes INSEE : 
    pool_communes = df_[(df_.neighbors_taux_de_couverture > moy_region)&(df_.Medecin_generaliste >= 1)].sort_values(by='neighbors_taux_de_couverture', ascending=False)
    itr_max = len(pool_communes)

    # Baseline avant lancement de l'itération :
    baseline = df_.neighbors_taux_de_couverture.mean()
    baseline_communes = len(df[df['neighbors_taux_de_couverture']<moy_region])

    # Listing des modifications :
    distance = []
    output = []
    recap = {}
    #trades = []
    #neighbors_stats = []

    # Début de l'itération
    for ind, row in pool_communes.iterrows():

        # Nombre de médecins disponibles AVANT transformation:
        med_dispo = nb_medecins_disponibles(df.loc[ind,:], selection_medecins=selection_medecins)

        # Identification des communes ayant une meilleure couverture que la moyenne de la région:
        if df_.loc[ind,'neighbors_taux_de_couverture'] > moy_region and med_dispo >= 1 :
            
            # Sélection des voisins, cleaning avec seulement les déficitaires:
            # Idée pour la suite: ne pourrait-on pas tester chaque type de 'sortby' et conserver le meilleur résultat ?
            temp_moy_region=moy_region # Valeur minimale, pouvant être repoussée si aucun voisin déficitaire n'est identifié
            temp_radius=radius

            neighbors_df = sort_neighbors(ind, rnc, df_, sortby=sortby, moy_region=temp_moy_region, radius=temp_radius, poids_des_voisins=poids_des_voisins)
            neighbors_df = neighbors_df[neighbors_df['neighbors_taux_de_couverture']<=temp_moy_region]
            count = len(neighbors_df)

            while count<nb_voisins_minimum:
                temp_moy_region+=0.05
                temp_radius+=1
                neighbors_df = sort_neighbors(ind, rnc, df_, sortby=sortby, moy_region=temp_moy_region, radius=temp_radius, poids_des_voisins=poids_des_voisins)
                neighbors_df = neighbors_df[neighbors_df['neighbors_taux_de_couverture']<=temp_moy_region]
                count = len(neighbors_df)

            #print(f"{count} voisins: {list(neighbors_df.index)}, distance_max: {temp_radius}")
            # Opération sur les nearest neighbors:
            transfer = False
            if count<= 0:
                print(f"Pas de voisin pour index {ind}")
            else:
                while transfer==False: 
                    ### Approche dispatch d'un seul médecin:
                    for key in list(neighbors_df.index):
                        count -= 1
                        medecins_assignables = -1 * int(neighbors_df.loc[key, 'medecins_assignables'])
                        #print(f"{med_dispo} à dispatcher, {medecins_assignables} médecins assignables")
                        
                        prev = df_.loc[key, 'Medecin_generaliste']
                        if medecins_assignables>=1 and med_dispo>0:
                            df_.loc[key, 'Medecin_generaliste'] = df_.loc[key, 'Medecin_generaliste'] + min(med_dispo,medecins_assignables)
                            df_.loc[ind, 'Medecin_generaliste'] = df_.loc[ind, 'Medecin_generaliste'] - min(med_dispo,medecins_assignables)
                            med_dispo -= min(med_dispo,medecins_assignables)
                            #trades.append([(df_.loc[ind,'code_insee'], df_.loc[key,'code_insee'], prev, df_.loc[key, 'Medecin_generaliste'])])
                            distance.append(haversine((df_.loc[key, 'Lat_commune'],df_.loc[key, 'Lon_commune']),(df.loc[ind, 'Lat_commune'],df.loc[ind, 'Lon_commune'])))
                        elif med_dispo>0:
                            df_.loc[key, 'Medecin_generaliste'] = df_.loc[key, 'Medecin_generaliste'] + 1
                            df_.loc[ind, 'Medecin_generaliste'] = df_.loc[ind, 'Medecin_generaliste'] - 1
                            med_dispo -= 1
                            #trades.append([(df_.loc[ind,'code_insee'], df_.loc[key,'code_insee'], prev, df_.loc[key, 'Medecin_generaliste'])])
                            distance.append(haversine((df_.loc[key, 'Lat_commune'],df_.loc[key, 'Lon_commune']),(df.loc[ind, 'Lat_commune'],df.loc[ind, 'Lon_commune'])))

                        if med_dispo <= 0:
                            transfer = True
                if recalcul==True:
                    itr+=1
                    df_ = preprocessing.get_meds_neighbors_df(df_)
                    if itr%10==0:
                        print(f'Traitement n°{itr}/{itr_max}')

    # Production du dictionnaire récapitulatif & chiffres clés
    df_ = preprocessing.get_meds_neighbors_df(df_)
    output = df_.neighbors_taux_de_couverture.mean()
    output_communes = len(df_[df_['neighbors_taux_de_couverture']<moy_region])

    #recap['trades'] = trades
    recap['distances'] = distance
    recap['scores'] = {
        'ancien_taux':round(baseline, 3),
        'nouveau_taux':round(output, 3),
        'delta_taux': round(output-baseline, 3),
        'ancien_communes_déficitaires': baseline_communes,
        'nouveau_communes_déficitaires': output_communes,
        'delta_communes': output_communes-baseline_communes,
        'distance_moyenne': np.array(distance).mean(), 
        'distance_min': np.array(distance).min(), 
        'distance_max': np.array(distance).max(),
        }
    #recap['nb_neighbors'] = neighbors_stats

    print(f"Statistiques médecins déplacés:\n - Moyenne: {np.array(distance).mean():.2f}\n - Min: {np.array(distance).min():.2f}\n - Max: {np.array(distance).max():.2f}")
    print(f"Médecins déplacés : {len(distance)}\n")
    print(f"Gain en points de: {(output - baseline)*100:.3f}%\navant : {baseline*100:.2f}% - après : {output.max()*100:.2f}%")
    print(f"Nb de communes en déficit avant : {baseline_communes} - après : {output_communes}\n")
    return df_[['code_insee', 'Medecin_generaliste', 'neighbors_taux_de_couverture']], recap


def nb_medecins_disponibles(row, selection_medecins='excédent'):
    if selection_medecins == 'excédent':
        if row.neighbors_nb_medecins-row.neighbors_Besoin_medecins < 0:
            return 0
        return min(int(row.Medecin_generaliste), int(row.neighbors_nb_medecins-row.neighbors_Besoin_medecins))
    elif selection_medecins == 'tous':
        return int(row.Medecin_generaliste)
    else:
        return 0

def sort_neighbors(ind, rnc, df_, sortby='distance', moy_region=0.84, radius=15, poids_des_voisins=0.1):
    '''
    Renvoie l'index de chaque voisin, avec 3 informations : 
        * Distance, en °
        * Le taux de couverture (avec voisinage)
        * Le déficit en médecins (avec voisinage) (valeur positive = manque de médecin, valeur négative = surplus de médecin)
    sortby peut prendre les valeurs suivantes : 'distance', 'deficit_rate', 'deficit_absolute', 'random', 'calculated'
    '''
    closest = rnc.radius_neighbors(X=[[df_.loc[ind,'Lat_commune'],df_.loc[ind,'Lon_commune']]],radius=radius*0.013276477888701137)
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
    neighbors_df=pd.DataFrame(data=neighbors_dict, index=['distance', 'neighbors_taux_de_couverture', 'neighbors_deficit_nb_medecins', 'local_deficit_nb_medecins', 'code_insee']).T
    if sortby=='distance':
        neighbors_df.sort_values(by='distance', inplace=True)
        # neighbors_dict={k: v for k, v in sorted(neighbors_dict.items(), key=lambda item: item[1][0])}
    elif sortby=='deficit_rate':
        neighbors_df.sort_values(by='neighbors_taux_de_couverture', inplace=True)
        # neighbors_dict={k: v for k, v in sorted(neighbors_dict.items(), key=lambda item: item[1][1])}
    elif sortby=='deficit_absolute':
        neighbors_df.sort_values(by=['local_deficit_nb_medecins', 'neighbors_deficit_nb_medecins'], inplace=True)
        # neighbors_dict={k: v for k, v in sorted(neighbors_dict.items(), key=lambda item: item[1][2])}
    elif sortby=='random':
        neighbors_df.sample(frac=1, )
    elif sortby=='calculated':
        neighbors_df.loc[:,'calculation']=neighbors_df.loc[:,'distance']*neighbors_df.loc[:,'neighbors_taux_de_couverture']+neighbors_df.loc[:,'local_deficit_nb_medecins']*0.1+neighbors_df.loc[:,'neighbors_deficit_nb_medecins']*0.03
        neighbors_df.sort_values(by='calculation', inplace=True)
    neighbors_df.loc[:,'medecins_assignables']=neighbors_df.loc[:,'local_deficit_nb_medecins']*1+neighbors_df.loc[:,'neighbors_deficit_nb_medecins']*poids_des_voisins

    return neighbors_df

'''
def algorithm_1_med(df, param='excédent'):
    df_ = df.copy()

    moy_region = 0.81

    #Initialisation des plus proches voisins : 
    rnc = NearestNeighbors(radius=15*0.013276477888701137, p=2)
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
            closest = rnc.radius_neighbors(X=[[df_.loc[ind,'Lat_commune'],df_.loc[ind,'Lon_commune']]],radius=15*0.013276477888701137)
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


def algorithm_all_available_med(df, selection_medecins='tous', sortby='rate', radius=15, moy_region=0.84, recalcul=True):
    
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
    
    df_ = df.copy()

    #Initialisation des plus proches voisins : 
    rnc = NearestNeighbors(radius=radius*0.013276477888701137, p=2)
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

class Medical_ReDispatch(BaseEstimator, ClassifierMixin):
    def __init__(self, selection_medecins='tous', sortby='calculated', radius=15, moy_region=0.84, recalcul=True, poids_des_voisins=0.1, nb_voisins_minimum=3, **kwargs):
        self.selection_medecins=selection_medecins
        self.sortby=sortby
        self.radius=radius
        self.moy_region=moy_region
        self.recalcul=recalcul
        self.poids_des_voisins=poids_des_voisins
        self.nb_voisins_minimum=nb_voisins_minimum
        #self.rnc = NearestNeighbors(radius=self.radius*0.013276477888701137, p=2)
        self.recap={}
        return None

    def fit(self, X, y=None, **fit_params):
        # Check that X has correct shape
        self.X_ = X
        var_radius = self.radius*0.013276477888701137
        rnc = NearestNeighbors(radius=var_radius, p=2)
        #Initialisation des plus proches voisins : 
        self.rnc = rnc.fit(self.X_[['Lat_commune', 'Lon_commune']])
         # Return the classifier
        return self

    def predict(self, X, y=None, ):
        # Check is fit had been called
        self.df_=X.copy()
        pool_communes = self.df_[(self.df_.neighbors_taux_de_couverture > self.moy_region)&(self.df_.Medecin_generaliste >= 1)].sort_values(by='neighbors_taux_de_couverture', ascending=False)
        self.baseline = self.df_.neighbors_taux_de_couverture.mean()
        self.baseline_communes = len(self.df_[self.df_['neighbors_taux_de_couverture']<self.moy_region])

        # Listing des modifications :
        self.distance = []
        self.output = []
        #trades = []
        #neighbors_stats = []

        # Début de l'itération
        for ind, row in pool_communes.iterrows():

            # Nombre de médecins disponibles AVANT transformation:
            med_dispo = nb_medecins_disponibles(self.df_.loc[ind,:], selection_medecins=self.selection_medecins)

            # Identification des communes ayant une meilleure couverture que la moyenne de la région:
            if self.df_.loc[ind,'neighbors_taux_de_couverture'] > self.moy_region and med_dispo >= 1 :
                
                # Sélection des voisins, cleaning avec seulement les déficitaires:
                # Idée pour la suite: ne pourrait-on pas tester chaque type de 'sortby' et conserver le meilleur résultat ?
                temp_moy_region=self.moy_region # Valeur minimale, pouvant être repoussée si aucun voisin déficitaire n'est identifié
                temp_radius=self.radius

                neighbors_df = sort_neighbors(ind, self.rnc, self.df_, sortby=self.sortby, moy_region=temp_moy_region, radius=temp_radius, poids_des_voisins=self.poids_des_voisins)
                neighbors_df = neighbors_df[neighbors_df['neighbors_taux_de_couverture']<=temp_moy_region]
                count = len(neighbors_df)

                while count<self.nb_voisins_minimum:
                    temp_moy_region+=0.05
                    temp_radius+=1
                    neighbors_df = sort_neighbors(ind, self.rnc, self.df_, sortby=self.sortby, moy_region=temp_moy_region, radius=temp_radius, poids_des_voisins=self.poids_des_voisins)
                    neighbors_df = neighbors_df[neighbors_df['neighbors_taux_de_couverture']<=temp_moy_region]
                    count = len(neighbors_df)

                #print(f"{count} voisins: {list(neighbors_df.index)}, distance_max: {temp_radius}")
                # Opération sur les nearest neighbors:
                transfer = False
                if count<= 0:
                    print(f"Pas de voisin pour index {ind}")
                else:
                    while transfer==False: 
                        ### Approche dispatch d'un seul médecin:
                        for key in list(neighbors_df.index):
                            count -= 1
                            medecins_assignables = -1 * int(neighbors_df.loc[key, 'medecins_assignables'])
                            #print(f"{med_dispo} à dispatcher, {medecins_assignables} médecins assignables")
                            
                            prev = self.df_.loc[key, 'Medecin_generaliste']
                            if medecins_assignables>=1 and med_dispo>0:
                                self.df_.loc[key, 'Medecin_generaliste'] = self.df_.loc[key, 'Medecin_generaliste'] + min(med_dispo,medecins_assignables)
                                self.df_.loc[ind, 'Medecin_generaliste'] = self.df_.loc[ind, 'Medecin_generaliste'] - min(med_dispo,medecins_assignables)
                                med_dispo -= min(med_dispo,medecins_assignables)
                                #trades.append([(df_.loc[ind,'code_insee'], df_.loc[key,'code_insee'], prev, df_.loc[key, 'Medecin_generaliste'])])
                                self.distance.append(haversine((self.df_.loc[key, 'Lat_commune'],self.df_.loc[key, 'Lon_commune']),(self.df_.loc[ind, 'Lat_commune'],self.df_.loc[ind, 'Lon_commune'])))
                            elif med_dispo>0:
                                self.df_.loc[key, 'Medecin_generaliste'] = self.df_.loc[key, 'Medecin_generaliste'] + 1
                                self.df_.loc[ind, 'Medecin_generaliste'] = self.df_.loc[ind, 'Medecin_generaliste'] - 1
                                med_dispo -= 1
                                #trades.append([(df_.loc[ind,'code_insee'], df_.loc[key,'code_insee'], prev, df_.loc[key, 'Medecin_generaliste'])])
                                self.distance.append(haversine((self.df_.loc[key, 'Lat_commune'],self.df_.loc[key, 'Lon_commune']),(self.df_.loc[ind, 'Lat_commune'],self.df_.loc[ind, 'Lon_commune'])))

                            if med_dispo <= 0:
                                transfer = True
                    if self.recalcul==True:
                        self.df_ = preprocessing.get_meds_neighbors_df(self.df_)

        # Production du dictionnaire récapitulatif & chiffres clés
        self.df_ = preprocessing.get_meds_neighbors_df(self.df_)
        self.output = self.df_.neighbors_taux_de_couverture.mean()
        self.output_communes = len(self.df_[self.df_['neighbors_taux_de_couverture']<self.moy_region])

        #recap['trades'] = trades
        self.distances = self.distance
        self.recap['ancien_taux']=round(self.baseline, 3)
        self.recap['nouveau_taux']=round(self.output, 3)
        self.recap['delta_taux']=round(self.output-self.baseline, 3)
        self.recap['ancien_communes_déficitaires']= self.baseline_communes
        self.recap['nouveau_communes_déficitaires']= self.output_communes
        self.recap['delta_communes']= self.output_communes-self.baseline_communes
        #self.recap['distance_moyenne']= np.array(distance).mean()
        #if len(distance)==0:
        #    self.recap['distance_min']= 'na'
        #    self.recap['distance_max']= 'na'
        #else:
        #    self.recap['distance_min']= np.array(distance).min()
        #    self.recap['distance_max']= np.array(distance).max()
        self.recap['médecins_déplacés']= len(self.distance)
        #recap['nb_neighbors'] = neighbors_stats

        #print(f"Statistiques médecins déplacés:\n - Moyenne: {np.array(distance).mean():.2f}\n - Min: {np.array(distance).min():.2f}\n - Max: {np.array(distance).max():.2f}")
        #print(f"Médecins déplacés : {len(distance)}\n")
        #print(f"Gain en points de: {(output - baseline)*100:.3f}%\navant : {baseline*100:.2f}% - après : {output.max()*100:.2f}%")
        #print(f"Nb de communes en déficit avant : {baseline_communes} - après : {output_communes}\n")
        return self.df_[['code_insee', 'Medecin_generaliste']]#, 'neighbors_taux_de_couverture']]#, recap
        #closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        #return self.y_[closest]

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {'selection_medecins':self.selection_medecins,
                'sortby':self.sortby,
                'radius':self.radius,
                'moy_region':self.moy_region,
                'recalcul':self.recalcul,
                'poids_des_voisins':self.poids_des_voisins,
                'nb_voisins_minimum':self.nb_voisins_minimum,
                }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def score(self, X):
        return self.recap
    
    def distance(self, X):
        return self.distance



'''