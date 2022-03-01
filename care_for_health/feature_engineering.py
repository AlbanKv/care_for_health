import pandas as pd
import numpy as np

def med_g_statistics():
    stats = {
    '18-24':[6, 21, 17, 22, 8, 11, 4, 9, 2],
    '25-34': [6, 24, 18, 16, 8, 9, 6, 9, 4],
    '35-49': [3, 24, 17, 16, 9, 12, 9, 8, 2],
    '50-64': [1, 13, 12, 11, 27, 10, 10, 12, 4],
    '65-': [1, 5, 15, 10, 29, 12, 13, 12, 3],
    }
    y_visits_med_g = {i: sum(np.array(stats[i])*0.01*np.array([0, 1, 2, 3, 4, 5, 6, 9.5, 14]).T) for i in stats}
    return y_visits_med_g

def yearly_med_g_visits(df, y_visits_med_g):
    # Création de la feature 'Nombre de visites annuelles de médecine générale'
    df['med_g_visites_annuelles'] = 0
    df.loc[:,'med_g_visites_annuelles'] = round(df.loc[:,'P18_POP0014']*3+\
                                    df.loc[:,'P18_POP1529']*y_visits_med_g['18-24']+\
                                    df.loc[:,'P18_POP3044']*y_visits_med_g['25-34']+\
                                    df.loc[:,'P18_POP4559']*y_visits_med_g['35-49']+\
                                    df.loc[:,'P18_POP6074']*y_visits_med_g['50-64']+\
                                    df.loc[:,'P18_POP7589']*y_visits_med_g['65-']+\
                                    df.loc[:,'P18_POP90P']*y_visits_med_g['65-'], 2)

    # Création de la feature 'Besoin en médecins généralistes'
    df['besoin_medecins_g'] = 0
    df.loc[:,'besoin_medecins_g'] = round(df.loc[:,'med_g_visites_annuelles']/3999, 2)
    
    return df
