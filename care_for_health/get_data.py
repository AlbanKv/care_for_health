import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from care_for_health import feature_engineering, preprocessing
from sklearn.neighbors import NearestNeighbors

def clean_data():
    """
    Supprime les médecins sans location
    """
    df = pd.read_csv("../raw_data/medecins_ge_fr.csv", delimiter=';', encoding='utf-8')
    
    df.dropna(subset=["code_insee"], inplace=True)
    
    # Correction erreur Sannerville
    df.loc[df["code_insee"] == "14666", "Département"] = 14.0
    df.loc[df["code_insee"] == "14666", "Code INSEE Région"] = 28.0
    
    df.reset_index(drop=True).to_csv("../raw_data/medecins_ge_fr_clean.csv", index=False)
    
    return df

def get_med():
    '''
    Import du fichier medecins_pdl présent dans raw_data
    '''
    med_cols={
    "Nature d'exercice": 'str',
    "Nom du professionnel": 'str',
    "Adresse": 'str',
    "Profession": 'str',
    "Coordonnées": 'str',
    "Commune": 'str',
    "code_insee": 'str',
    "Département": 'str',
    "Code INSEE Région": "str"
    } 
    df = pd.read_csv("../raw_data/medecins_ge_fr_clean.csv", delimiter=',', encoding='utf-8', usecols=list(med_cols.keys()), dtype=med_cols)
    
    #df avec seulement les colonnes utiles :
    #'Nom du professionnel', 'Profession', "Nature d'exercice",
    # 'Coordonnées', 'Adresse', 'Commune','Département', 'code_insee'
    df_2 = df[['Nom du professionnel', 'Profession', "Nature d'exercice",'Coordonnées', 'Adresse', 'Commune','Département', 'code_insee', "Code INSEE Région"]].copy()
    #cleaning doublons
    df_3= df_2.drop_duplicates().reset_index(drop=True).copy()
    # séparation des Coordonnées en Latitude, Longitude
    df_3[['Lat','Long']] = df_3["Coordonnées"].str.split(",",expand=True)
    #dataframe prêt
    medecins = df_3[['Nom du professionnel', 'Profession', "Nature d'exercice",'Lat','Long', 'Adresse', 'Commune','Département', 'code_insee', "Code INSEE Région"]].copy()

    return medecins
#['44', '49', '53', '72', '85']
def read_base_insee(cp_list= None):
    colonnes = {
            'CODGEO': 'str', 
            'P18_POP': 'float', 
            'P18_POP0014': 'float', 
            'P18_POP1529': 'float', 
            'P18_POP3044': 'float',
            'P18_POP4559': 'float', 
            'P18_POP6074': 'float', 
            'P18_POP7589': 'float',
            'P18_POP90P': 'float', 
            'DECE1318': 'float',
            'NAIS1318': 'float', 
            'C18_POP55P_CS7': 'float', 
            'P13_POP': 'float', 
            'MED19': 'float', 
            'TP6019': 'float'
            }
    df_insee = pd.read_csv('../raw_data/base_insee.csv', delimiter=';', encoding='utf-8',  usecols=list(colonnes.keys()), dtype=colonnes)
    df_insee = df_insee[['CODGEO', 'P18_POP','P18_POP0014', 'P18_POP1529', 'P18_POP3044','P18_POP4559','P18_POP6074','P18_POP7589','P18_POP90P','DECE1318','NAIS1318','C18_POP55P_CS7','P13_POP', 'MED19', 'TP6019',]].copy()
    df_insee['CODGEO'] = df_insee['CODGEO'].astype(str)
 
    """df_insee = df_insee[(df_insee['CODGEO'].astype(str).str.startswith(cp)==True)|\
            (df_insee['CODGEO'].astype(str).str.startswith(cp_pdl[1])==True)|\
            (df_insee['CODGEO'].astype(str).str.startswith(cp_pdl[2])==True)|\
            (df_insee['CODGEO'].astype(str).str.startswith(cp_pdl[3])==True)|\
            (df_insee['CODGEO'].astype(str).str.startswith(cp_pdl[4])==True)].copy()"""
    df_insee = df_insee.loc[df_insee['CODGEO'].astype(str).str.len()==5].copy()
    return df_insee


def merge_insee_med():
    '''
    Merge les datasets des médecins_pdl avec le dataset insee afin de récupérer les coordonnées géographiques
    '''
    df_insee = read_base_insee()
    df_medecins = get_med()
    
    # Merge 
    # On supprime la colonne CODGEO car doublon de code_insee
    df_merge = df_insee.merge(df_medecins, left_on="CODGEO", right_on="code_insee").drop(columns="CODGEO")
    
    # Placer la colonne code_insee en premier pour simplification d'affichage
    first_column = df_merge.pop('code_insee')
    df_merge.insert(0, 'code_insee', first_column)
    second_col = df_merge.pop('Code INSEE Région')
    df_merge.insert(1, 'Code INSEE Région', second_col)
    
    return df_merge


def get_full_medbase(region=None):
    """
    region = liste des régions à selectionner
             par défaut séléctionne toutes les régions de France métropole
             
    region_dict = {'84':"Auvergne-Rhône-Alpes",
                   '27':"Bourgogne-Franche-Comté",
                   '53':"Bretagne",
                   '24':"Centre-Val de Loire",
                   '94':"Corse",
                   '44':"Grand Est",
                   '32':"Hauts-de-France",
                   '11':"Île-de-France",
                   '28':"Normandie",
                   '75':"Nouvelle-Aquitaine",
                   '76':"Occitanie",
                   '52':"Pays de la Loire",
                   '93':"Provence-Alpes-Côte d'Azur",}
    """
    
    try:
        pd.read_csv("../raw_data/medecins_ge_fr_clean.csv", delimiter=',', encoding='utf-8')
            
    except FileNotFoundError:
        clean_data()
    
    df = merge_insee_med()
    col_val = ['Médecin généraliste']#, 'Chirurgien-dentiste', 'Radiologue', 'Sage-femme', 'Ophtalmologiste', 'Cardiologue']
    #short_df = df[df['Profession'].isin(col_val)].copy()
    #communes avec leurs polygon (affichage de map)
    df_comm = pd.read_csv("../raw_data/communes_fr.csv", delimiter=',', encoding='utf-8')[["codgeo", "geometry"]]
    df_insee = read_base_insee()
    #long/lat des communes
    df_gps_comm = pd.read_csv('../raw_data/communes_gps.csv', delimiter=',', encoding='utf-8')[["code_commune_INSEE", "latitude", "longitude"]]
    
    # Récupération des communes pdl (polygon)
    """cp_pdl = ['44', '49', '53', '72', '85']
    df_comm_pdl = df_comm[(df_comm["codgeo"].astype(str).str.startswith(cp_pdl[0])==True) |\
                        (df_comm['codgeo'].astype(str).str.startswith(cp_pdl[1])==True)|\
                        (df_comm['codgeo'].astype(str).str.startswith(cp_pdl[2])==True)|\
                        (df_comm['codgeo'].astype(str).str.startswith(cp_pdl[3])==True)|\
                        (df_comm['codgeo'].astype(str).str.startswith(cp_pdl[4])==True)].copy()"""

    #OneHotEncode on selected professions:
    """ encoder = OneHotEncoder(sparse=False)
    encoder.fit(short_df[['Profession']])
    profession_encoded = encoder.transform(short_df[['Profession']])
    enc = encoder.categories_[0]
    short_df[enc[0]], short_df[enc[1]], short_df[enc[2]], short_df[enc[3]], short_df[enc[4]], short_df[enc[5]] = profession_encoded.T
    """
    # merge des communes sans médecins et ajout des polygon
    #df_merge = df_comm_pdl.merge(short_df, how="left", left_on="codgeo", right_on="code_insee")#.drop(columns="codgeo")
    df_merge = df_comm.merge(df, how="left", left_on="codgeo", right_on="code_insee")

    # empêcher la création de colonnes en double
    same_cols = [col for col in df_insee.columns if col in df_merge.columns]
    
    # ajout des infos insee pour les communes sans médecins
    df_merge = df_merge.drop(columns=same_cols).merge(df_insee, how="left", left_on="codgeo", right_on="CODGEO")
    
    # suppression de la colonne codegeo car doublon de code_insee
    df_merge["code_insee"] = df_merge["codgeo"]
    df_merge.drop(columns="codgeo", inplace=True)
    
    # Placer la colonne code_insee en premier pour simplification d'affichage
    first_column = df_merge.pop('code_insee')
    df_merge.insert(0, 'code_insee', first_column)
    
    # récupérer les coordonnées gps des communes
    """df_gps_comm = df_gps_comm[((df_gps_comm["code_commune_INSEE"].str.startswith(cp_pdl[0])==True) |\
                                        (df_gps_comm['code_commune_INSEE'].str.startswith(cp_pdl[1])==True)|\
                                        (df_gps_comm['code_commune_INSEE'].str.startswith(cp_pdl[2])==True)|\
                                        (df_gps_comm['code_commune_INSEE'].str.startswith(cp_pdl[3])==True)|\
                                        (df_gps_comm['code_commune_INSEE'].str.startswith(cp_pdl[4])==True)) & \
                                        (df_gps_comm['code_commune_INSEE'].apply(len) == 5)].copy().drop_duplicates().reset_index(drop=True)"""
    df_gps_comm = df_gps_comm.drop_duplicates().reset_index(drop=True)
    df_gps_comm.columns = ["code_insee", "Lat_commune", "Lon_commune"]

    df_merge = df_merge.merge(df_gps_comm, on="code_insee")

    #Transform lat, lon in float
    df_merge['Lat'] = df_merge['Lat'].astype(float)
    df_merge['Long'] = df_merge['Long'].astype(float)
    
    
    df_merge.rename(columns={"Code INSEE Région": "code_regions"}, inplace=True)
    
    df_corres = df_merge[["code_insee", "code_regions"]].drop_duplicates().dropna()
    df_corres.rename(columns={"code_regions": "code_reg"}, inplace=True)
    df_corres["code_dep"] = df_corres["code_insee"].apply(lambda x: x[:2])
    df_corres = df_corres.drop(columns="code_insee").drop_duplicates().reset_index(drop=True)
    
    df_merge["code_dpt"] = df_merge["code_insee"].apply(lambda x: x[:2])
    df_merge["code_regions"] = df_merge.merge(df_corres, left_on="code_dpt", right_on="code_dep")["code_reg"].astype(float).astype(int)

    # par défaut : régions de métropole
    if not region:
        region = [11,24,27,28,32,44,52,53,75,76,84,93,94]
    df_merge = df_merge[df_merge["code_regions"].isin(region)]

    #Prepare dataset with stats for 'médecin généraliste'
    stats = feature_engineering.med_g_statistics()
    df_merge = feature_engineering.yearly_med_g_visits(df_merge, stats)

    #Aggregate rows together:
    # Merge pour ajouter la colonne geometry sans aggrégat
    prof_df = df_merge[["code_insee", "code_regions", "code_dpt", "geometry", "Lat_commune", "Lon_commune"]].merge(df_merge.groupby('code_insee', as_index=False).agg(
        Population_2018=('P18_POP','mean'), 
        Deces_13_18= ('DECE1318','mean'),
        Naissances_13_18=('NAIS1318','mean'),
        Retraites_2018_55p=('C18_POP55P_CS7','mean'), 
        Population_2013=('P13_POP','mean'), 
        Mediane_revenu=('MED19','mean'), 
        Taux_pauvrete=('TP6019','mean'), 
        Besoin_annuel_visites_med_g=('med_g_visites_annuelles', 'mean'),
        Besoin_medecins=('besoin_medecins_g', 'mean'),
        Medecin_generaliste=('Profession','count'), 
        ), how="left", on="code_insee").drop_duplicates().reset_index(drop=True)
    
    prof_df['taux_de_couverture']= prof_df['Medecin_generaliste']/prof_df['Besoin_medecins']
    
    prof_df[prof_df.select_dtypes(include=np.number).columns.tolist()] = prof_df.select_dtypes(include=np.number).apply(lambda x: np.round(x, 2))

    #correction du NaN pour les communes sans besoin de médecins
    prof_df.loc[prof_df["Besoin_medecins"] == 0, "taux_de_couverture"] = 0
    
    return prof_df


def get_data_region():
    df = get_full_medbase()
    df_ = df.groupby('code_regions', as_index=False).agg(
            Population_2018=('Population_2018','sum'),
            Deces_13_18= ('Deces_13_18','sum'),
            Naissances_13_18=('Naissances_13_18','sum'),
            Retraites_2018_55p=('Retraites_2018_55p','sum'),
            Population_2013=('Population_2013','sum'),
            Mediane_revenu=('Mediane_revenu','mean'),
            Taux_pauvrete=('Taux_pauvrete','mean'),
            Besoin_annuel_visites_med_g=('Besoin_annuel_visites_med_g', 'sum'),
            Besoin_medecins=('Besoin_medecins', 'sum'),
            Medecin_generaliste=('Medecin_generaliste','sum'),
            taux_de_couverture=('taux_de_couverture','mean'),
            ).drop_duplicates().reset_index(drop=True)
    df_['taux_de_couverture']=0
    df_['taux_de_couverture']=round(df_['Medecin_generaliste']/df_['Besoin_medecins'],2)
    
    df_[df_.select_dtypes(include=np.number).columns.tolist()] = df_.select_dtypes(include=np.number).apply(lambda x: np.round(x, 2))
    
    return df_

def get_full_medbase_with_neighbors(radius=30, reduce_column_nb=True):
    '''
    Builds up the full medbase along with neighbors + cover rate for the neighbors population
    '''
    df = get_full_medbase()
    df = preprocessing.list_neighbors_by_df(df, radius=radius)
    df = preprocessing.get_meds_neighbors_df(df)
    if reduce_column_nb==True:
        df_ = df[['code_insee', 'geometry', 'Lat_commune', 'Lon_commune', 'Population_2018', 
            'Besoin_annuel_visites_med_g', 'Besoin_medecins', 'Medecin_generaliste', 'taux_de_couverture', 
            'neighbors', 'neighbors_Besoin_medecins', 'neighbors_nb_medecins', 'neighbors_taux_de_couverture']].copy()
    else:
        df_ = df.copy()
    return df_

def get_all_neighbors_by_csv(radius=1, name=None):
    if not name:
            name = "../raw_data/communes_neighbors.csv"
    
    try:
        df_comms_neighbors = pd.read_csv(name, delimiter=',', encoding='utf-8')
            
    except FileNotFoundError:
        create_dataframe_neighbor(radius, name)
        
        df_comms_neighbors = pd.read_csv(name, delimiter=',', encoding='utf-8')
        
    return df_comms_neighbors

def get_all_neighbors(radius=1, name=None):
    df_base = get_full_medbase()
    df_comms_neighbors = pd.DataFrame(columns=["code_insee", "neighbors_code_insee", "distance", "taux_de_couverture"])
        
    rnc = NearestNeighbors(radius=radius, p=2)
    rnc.fit(df_base[['Lat_commune', 'Lon_commune']])
        
    for index, row_base in df_base.iterrows():
        closest = rnc.radius_neighbors(X=[[row_base.get('Lat_commune'), row_base.get('Lon_commune')]],radius=radius/80*8)
            
        # ignore le premier résultat qui est la commune elle-même
        for i in range(1, len(closest[0][0])):
            row_neighbor = df_base.loc[closest[1][0][i]]
                
            df_comms_neighbors.loc[len(df_comms_neighbors)] = {"code_insee":row_base.get("code_insee"), 
                                    "neighbors_code_insee": row_neighbor.get("code_insee"),
                                    "distance": closest[0][0][i], 
                                    "taux_de_couverture": row_neighbor.get('taux_de_couverture')}
            
    return df_comms_neighbors

def create_dataframe_neighbor(radius=1, name=None):
    if not name:
            name = "../raw_data/communes_neighbors.csv"
            
    get_all_neighbors(radius).to_csv(name, index=False)
    
'''
Deprecated function
'''
'''def read_pdl():
    pdl = pd.read_csv('../raw_data/pays_de_la_loire.csv', delimiter=';', encoding='utf-8')
    pdl.DATE_DEBUT_VALIDITE = pd.to_datetime(pdl.DATE_DEBUT_VALIDITE)
    base_med_pays_de_la_loire = pdl[['CODE_DEPARTEMENT', 'LIBELLE_PROFESSION', 'LIBELLE_MODE_EXERCICE', 'RAISON_SOCIALE_SITE', 'IDENTIFIANT_PP', 'LIBELLE_COMMUNE_COORD_STRUCTURE','CODE_POSTAL_COORD_STRUCTURE', 'CODE_COMMUNE_COORD_STRUCTURE']].copy()
    return base_med_pays_de_la_loire

def get_insee():
    laposte = read_laposte()
    base_insee = read_base_insee()
    base_insee['CODGEO'] = base_insee['CODGEO'].astype(str)
    laposte.loc[:,'code_commune_insee'] = laposte.loc[:,'code_commune_insee'].astype(str)
    df = pd.merge(left=base_insee, right=laposte, left_on='CODGEO', right_on='code_commune_insee', how='left')
    nb_row_insee = len(base_insee)
    nb_row_df = len(df)
    print(f"Attention : un tri doit encore être réalisé sur les codes postaux / codes commune INSEE car {nb_row_df - nb_row_insee} doublons sont apparus sur {nb_row_df} lors du merge")
    print("Il y a actuellement un merge avec le code postal. Si celui-ci n'est plus requis, ne pas tenir compte du message précédent")

    return df

def read_laposte():
    lp1 = pd.read_csv('../raw_data/laposte_hexasmal.csv', delimiter=';', encoding='utf-8')
    lp1 = lp1.drop_duplicates()
    # Drop des codes CORSE
    lp_1 = lp1.drop((lp1[lp1['code_commune_insee'].astype(str).str.startswith('2A')==True]).index).copy()
    lp_2 = lp_1.drop((lp_1[lp_1['code_commune_insee'].astype(str).str.startswith('2B')==True]).index).copy()    
    return lp_2[['code_postal', 'code_commune_insee', 'nom_de_la_commune']].copy()


'''