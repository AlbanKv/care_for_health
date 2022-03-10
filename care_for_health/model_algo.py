from haversine import haversine, Unit 
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
import itertools

from care_for_health import preprocessing
from care_for_health.algorithm import nb_medecins_disponibles, sort_neighbors

import numpy as np
import pandas as pd
import random


class Medical_ReDispatch(BaseEstimator, ClassifierMixin):
    def __init__(self, selection_medecins='tous', sortby='calculated', radius=15, moy_region=None, recalcul=False, poids_des_voisins=0.1, nb_voisins_minimum=3, **kwargs):
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
        self.baseline = self.X_.neighbors_taux_de_couverture.mean()
        self.baseline_communes = len(self.X_[self.X_['neighbors_taux_de_couverture']<self.moy_region])
        var_radius = self.radius*0.013276477888701137
        rnc = NearestNeighbors(radius=var_radius, p=2)
        #Initialisation des plus proches voisins : 
        self.rnc = rnc.fit(self.X_[['Lat_commune', 'Lon_commune']])
         # Return the classifier
        self.is_fitted_ = True
        self.recap = {'delta_taux': 0}
        return self

    def predict(self, X, y=None, ):
        # Check is fit had been called
        self.df_=X.copy()
        pool_communes = self.df_[(self.df_.neighbors_taux_de_couverture > self.moy_region)&(self.df_.Medecin_generaliste >= 1)].sort_values(by='neighbors_taux_de_couverture', ascending=False)

        # Listing des modifications :
        self.distance = []
        self.output = []
        #trades = []
        #neighbors_stats = []
        
        if self.moy_region:
            df_["moy_region"] = self.moy_region
        else:
            df_moy_reg = df_.groupby("code_regions", as_index=False).agg(sum_besoin=("Besoin_medecins", "sum"), sum_meds=("Medecin_generaliste", "sum"))
            df_moy_reg["moy_region"] = round(df_moy_reg["sum_meds"] / df_moy_reg["sum_besoin"], 2)

            df_moy_reg.drop(columns=["sum_besoin", "sum_meds"], inplace=True)

            df_ = df_.merge(df_moy_reg, how="left", on="code_regions")

        # Début de l'itération
        for ind, row in pool_communes.iterrows():

            # Nombre de médecins disponibles AVANT transformation:
            med_dispo = nb_medecins_disponibles(self.df_.loc[ind,:], selection_medecins=self.selection_medecins)

            # Identification des communes ayant une meilleure couverture que la moyenne de la région:
            if self.df_.loc[ind,'neighbors_taux_de_couverture'] > df_.moy_region and med_dispo >= 1 :
                
                # Sélection des voisins, cleaning avec seulement les déficitaires:
                # Idée pour la suite: ne pourrait-on pas tester chaque type de 'sortby' et conserver le meilleur résultat ?
                temp_moy_region=df_.moy_region # Valeur minimale, pouvant être repoussée si aucun voisin déficitaire n'est identifié
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
        self.output_communes = len(self.df_[self.df_['neighbors_taux_de_couverture']<df_.moy_region])

        #recap['trades'] = trades
        self.distances = self.distance
        self.recap['ancien_taux']=round(self.baseline, 3)
        self.recap['nouveau_taux']=round(self.output, 3)
        self.recap['delta_taux']=round(self.output-self.baseline, 3)
        self.recap['ancien_communes_déficitaires']= self.baseline_communes
        self.recap['nouveau_communes_déficitaires']= self.output_communes
        self.recap['delta_communes']= self.output_communes-self.baseline_communes
        self.recap['distance_moyenne']= np.array(self.distance).mean()
        if len(self.distance)==0:
            self.recap['distance_min']= 'na'
            self.recap['distance_max']= 'na'
        else:
            self.recap['distance_min']= np.array(self.distance).min()
            self.recap['distance_max']= np.array(self.distance).max()
        self.recap['médecins_déplacés']= len(self.distance)

        return self.df_[['code_insee', 'neighbors_taux_de_couverture']]#, 'neighbors_taux_de_couverture']]#, recap

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
    
    def score(self, X, *args, **kwargs):
        return self.recap
    
    def distance(self, X):
        return self.distance


def grid_search_med(X, **kwargs):
    '''
    Passe une liste des paramètres à tester (voir la forme à suivre) et renvoie un dataframe contenant les principaux scores : 
    
    params=dict(
        selection_medecins=['tous', 'excédent'],
        sortby=['distance', 'deficit_rate', 'deficit_absolute', 'random', 'calculated'],
        radius=[10, 15, 20],
        moy_region=[0.82],
        recalcul=[True, False],
        poids_des_voisins=[0.1, 0.4],#coefficient appliqué au calcul du nombre de médecins requis
        nb_voisins_minimum=[2, 3, 4],#élargissement du pool de voisins, si les voisins sont déjà bien dotés et/ou trop peu nombreux
        )
    '''
    if kwargs:
        if len(kwargs)!=7:
            return "7 colonnes sont attendues"

        # Calcul du nombre de combinaisons attendues :
        produit=1
        columns = [str(k) for k in kwargs.keys()]
        for k, v in kwargs.items():
            if isinstance(v, list):
                produit = produit*len(v)
        stack=[]
        for k,v in kwargs.items():
            temp=[]
            for elem in v:
                temp.append(elem)
            stack.append(temp)
        all_combinations = list(itertools.product(stack[0], stack[1], stack[2], stack[3], stack[4], stack[5], stack [6]))

        # Création de la grille de tests
        grid=pd.DataFrame(columns=columns, data=all_combinations)#[range(36)])#data=[kwargs.values()])
        print(f'forme de la grille à tester: {grid.shape}')
        
        # Création de la grille de résultats
        grid_results = pd.DataFrame(columns=['ancien_taux', 'nouveau_taux', 'delta_taux', 'ancien_communes_déficitaires', 'nouveau_communes_déficitaires', 'delta_communes', 'médecins_déplacés', 'distance_min', 'distance_max', 'distance_moyenne', 'params'])
        n=0
        
        # Initialisation du modèle
        model = Medical_ReDispatch()

        # Conduite du grid_search
        for ind, row in grid.iterrows():

            #initialisation du modèle
            testing_params = dict(grid.loc[ind,:])
            model.set_params(**testing_params)
            model.fit(X)
            model.predict(X)
            scr=model.score(X)
            print(scr)
            # Enregistrement des résultats
            grid_results.loc[n,:]=[
                scr['ancien_taux'],
                scr['nouveau_taux'], 
                scr['delta_taux'], 
                scr['ancien_communes_déficitaires'],
                scr['nouveau_communes_déficitaires'],
                scr['delta_communes'],
                scr['médecins_déplacés'],
                scr['distance_min'],
                scr['distance_max'],
                scr['distance_moyenne'],
                testing_params,
            ]
            n+=1
            print(f'{n} modeles testés')

        return grid_results

'''
class Medical_ReDispatch_v3(BaseEstimator, ClassifierMixin):
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
        self.df_ = X
        self.baseline = self.df_.neighbors_taux_de_couverture.mean()
        self.baseline_communes = len(self.df_[self.df_['neighbors_taux_de_couverture']<self.moy_region])
        var_radius = self.radius*0.013276477888701137
        rnc = NearestNeighbors(radius=var_radius, p=2)
        #Initialisation des plus proches voisins : 
        self.rnc = rnc.fit(self.df_[['Lat_commune', 'Lon_commune']])
         # Return the classifier
        self.is_fitted_ = True
        self.recap = {'delta_taux': 0}
        self.score = 0

        pool_communes = self.df_[(self.df_.neighbors_taux_de_couverture > self.moy_region)&(self.df_.Medecin_generaliste >= 1)].sort_values(by='neighbors_taux_de_couverture', ascending=False)

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
        self.score = self.output-self.baseline

        #recap['trades'] = trades
        self.distances = self.distance
        self.recap['ancien_taux']=round(self.baseline, 3)
        self.recap['nouveau_taux']=round(self.output, 3)
        self.recap['delta_taux']=round(self.output-self.baseline, 3)
        self.recap['ancien_communes_déficitaires']= self.baseline_communes
        self.recap['nouveau_communes_déficitaires']= self.output_communes
        self.recap['delta_communes']= self.output_communes-self.baseline_communes
        self.recap['distance_moyenne']= np.array(self.distance).mean()
        if len(self.distance)==0:
            self.recap['distance_min']= 'na'
            self.recap['distance_max']= 'na'
        else:
            self.recap['distance_min']= np.array(self.distance).min()
            self.recap['distance_max']= np.array(self.distance).max()
        self.recap['médecins_déplacés']= len(self.distance)

        return self

    def predict(self, X, y=None, ):
        return self.df_[['code_insee', 'Medecin_generaliste']]#, 'neighbors_taux_de_couverture']]#, recap

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

    def score(self, X, *args, **kwargs):
        return self.score

    def distance(self, X):
        return self.distance
'''