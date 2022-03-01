import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from care_for_health import feature_engineering

def get_med_pdl():
    #import du fichier medecins_pdl présent dans raw_data
    df = pd.read_csv("../raw_data/medecins_pdl.csv", delimiter=';', encoding='utf-8')
    #df avec seulement les colonnes utiles :
    #'Nom du professionnel', 'Profession', "Nature d'exercice",
    # 'Coordonnées', 'Adresse', 'Commune','Département', 'code_insee'
    df_2 = df[['Nom du professionnel', 'Profession', "Nature d'exercice",'Coordonnées', 'Adresse', 'Commune','Département', 'code_insee']].copy()
    #cleaning doublons
    df_3= df_2.drop_duplicates().copy()
    # séparation des Coordonnées en Latitude, Longitude
    df_3[['Lat','Long']] = df_3["Coordonnées"].str.split(",",expand=True)
    #dataframe prêt
    medecins_pdl = df_3[['Nom du professionnel', 'Profession', "Nature d'exercice",'Lat','Long', 'Adresse', 'Commune','Département', 'code_insee']].copy()
    medecins_pdl["code_insee"] = medecins_pdl["code_insee"].astype(str)

    return medecins_pdl

def read_laposte():
    lp1 = pd.read_csv('../raw_data/laposte_hexasmal.csv', delimiter=';', encoding='utf-8')
    lp1 = lp1.drop_duplicates()
    # Drop des codes CORSE
    lp_1 = lp1.drop((lp1[lp1['code_commune_insee'].astype(str).str.startswith('2A')==True]).index).copy()
    lp_2 = lp_1.drop((lp_1[lp_1['code_commune_insee'].astype(str).str.startswith('2B')==True]).index).copy()    
    return lp_2[['code_postal', 'code_commune_insee', 'nom_de_la_commune']].copy()

def read_base_insee():
    cod_var_retenus = [
    'CODGEO',
    'P18_POP',
    'P18_POP0014',
    'P18_POP1529',
    'P18_POP3044',
    'P18_POP4559',
    'P18_POP6074',
    'P18_POP7589',
    'P18_POP90P',
    'DECE1318',
    'NAIS1318',
    'C18_POP55P_CS7',
    'P13_POP',
    'MED19',
    'TP6019',
    ]
    df_insee = pd.read_csv('../raw_data/base_insee.csv', delimiter=';', encoding='utf-8',  usecols=cod_var_retenus)
    df_insee = df_insee[['CODGEO', 'P18_POP','P18_POP0014', 'P18_POP1529', 'P18_POP3044','P18_POP4559','P18_POP6074','P18_POP7589','P18_POP90P','DECE1318','NAIS1318','C18_POP55P_CS7','P13_POP', 'MED19', 'TP6019',]].copy()
    cp_pdl = ['44', '49', '53', '72', '85']
    df_insee['CODGEO'] = df_insee['CODGEO'].astype(str)
    df_insee = df_insee[(df_insee['CODGEO'].astype(str).str.startswith(cp_pdl[0])==True)|\
            (df_insee['CODGEO'].astype(str).str.startswith(cp_pdl[1])==True)|\
            (df_insee['CODGEO'].astype(str).str.startswith(cp_pdl[2])==True)|\
            (df_insee['CODGEO'].astype(str).str.startswith(cp_pdl[3])==True)|\
            (df_insee['CODGEO'].astype(str).str.startswith(cp_pdl[4])==True)].copy()
    df_insee = df_insee.loc[df_insee['CODGEO'].astype(str).str.len()==5].copy()
    return df_insee


def merge_insee_med():
    '''Merge les datasets des médecins_pdl avec le dataset insee afin de récupérer les coordonnées géographiques'''
    df_insee = read_base_insee()
    df_medecins = get_med_pdl()
    
    # Merge 
    # On supprime la colonne CODGEO car doublon de code_insee
    df_merge = df_insee.merge(df_medecins, left_on="CODGEO", right_on="code_insee").drop(columns="CODGEO")
    
    # Placer la colonne code_insee en premier pour simplification d'affichage
    cols = list(df_merge.columns)
    index = df_merge.columns.get_loc("code_insee")
    cols = [cols[index]] + cols[:index]
    df_merge = df_merge[cols]
    
    return df_merge

def get_full_medbase():
    df = merge_insee_med()
    col_val = ['Médecin généraliste', 'Chirurgien-dentiste', 'Radiologue', 'Sage-femme', 'Ophtalmologiste', 'Cardiologue']
    short_df = df[df['Profession'].isin(col_val)].copy()

    #OneHotEncode on selected professions:
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(short_df[['Profession']])
    profession_encoded = encoder.transform(short_df[['Profession']])
    enc = encoder.categories_[0]
    short_df[enc[0]], short_df[enc[1]], short_df[enc[2]], short_df[enc[3]], short_df[enc[4]], short_df[enc[5]] = profession_encoded.T

    #Transform lat, lon in float
    short_df['Lat'] = short_df['Lat'].astype(float)
    short_df['Long'] = short_df['Long'].astype(float)

    #Prepare dataset with stats for 'médecin généraliste'
    stats = feature_engineering.med_g_statistics()
    short_df = feature_engineering.yearly_med_g_visits(short_df, stats)

    #Aggregate rows together:
    prof_df = short_df.groupby('code_insee', as_index=False).agg(
        Population_2018=('P18_POP','mean'), 
        Décès_13_18= ('DECE1318','mean'),
        Naissances_13_18=('NAIS1318','mean'),
        Retraités_2018_55p=('C18_POP55P_CS7','mean'), 
        Population_2013=('P13_POP','mean'), 
        Médiane_revenu=('MED19','mean'), 
        Taux_pauvreté=('TP6019','mean'), 
        Lat=('Lat','mean'), 
        Lon=('Long','mean'), 
        Besoin_annuel_visites_méd_g=('med_g_visites_annuelles', 'mean'),
        Besoin_médecins=('besoin_medecins_g', 'mean'),
        Médecin_généraliste=('Médecin généraliste','sum'), 
        Cardiologue=('Cardiologue','sum'), 
        Chirurgien_dentiste=('Chirurgien-dentiste','sum'), 
        Ophtalmologiste=('Ophtalmologiste','sum'), 
        Radiologue=('Radiologue','sum'), 
        Sage_femme=('Sage-femme','sum'), 
        )
    return prof_df


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

def read_pdl():
    pdl = pd.read_csv('../raw_data/pays_de_la_loire.csv', delimiter=';', encoding='utf-8')
    pdl.DATE_DEBUT_VALIDITE = pd.to_datetime(pdl.DATE_DEBUT_VALIDITE)
    base_med_pays_de_la_loire = pdl[['CODE_DEPARTEMENT', 'LIBELLE_PROFESSION', 'LIBELLE_MODE_EXERCICE', 'RAISON_SOCIALE_SITE', 'IDENTIFIANT_PP', 'LIBELLE_COMMUNE_COORD_STRUCTURE','CODE_POSTAL_COORD_STRUCTURE', 'CODE_COMMUNE_COORD_STRUCTURE']].copy()
    return base_med_pays_de_la_loire

'''