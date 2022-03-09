from pyexpat import model
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime
import pytz
import joblib
from care_for_health import model_algo
import ast
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"Salutations": "Coucou le monde"}

@app.get("/predict")
def predict(
    selection_medecins='tous',
    sortby='calculated',
    radius=15,
    moy_region=0.84,
    recalcul=False,
    poids_des_voisins=0.1,
    nb_voisins_minimum=2,
    ):

    params=dict(
        selection_medecins=selection_medecins,
        sortby=sortby,
        radius=int(radius),
        moy_region=float(moy_region),
        recalcul=recalcul,
        poids_des_voisins=float(poids_des_voisins),#coefficient appliqué au calcul du nombre de médecins requis
        nb_voisins_minimum=int(nb_voisins_minimum),#élargissement du pool de voisins, si les voisins sont déjà bien dotés et/ou trop peu nombreux
        )
    
    df = pd.read_csv('brouillon/df_api_test.csv', delimiter=',', dtype={'code_insee':'str'}, converters={"neighbors": lambda x: ast.literal_eval(x)}).reindex()

    model = model_algo.Medical_ReDispatch()
    model.set_params(**params)
    model.fit(df)
    y_pred = model.predict(df)
    y_dict = y_pred.set_index('code_insee').to_dict()
    
    scr = model.score(df)
    return {
        'Ancien_taux': scr['ancien_taux'],
        'Nouveau_taux': scr['nouveau_taux'],
        'Evolution du taux': scr['delta_taux'],
        "Communes n'étant plus déficitaires": scr['delta_communes'],
        'distance_moyenne_parcourue': scr['distance_moyenne'],
        'distance_totale_parcourue': f"{scr['distance_moyenne']*scr['médecins_déplacés']:.2f}",
        'data':json.dumps(y_dict)
    }

'''
        }
    scr['nouveau_taux'], 
    scr['delta_taux'], 
    scr['ancien_communes_déficitaires'],
    scr['nouveau_communes_déficitaires'],
    scr['delta_communes'],
    scr['médecins_déplacés'],
    scr['distance_min'],
    scr['distance_max'],
    scr['distance_moyenne'],


    return {'prediction': 'toutvabien'}#model.predict(df)}

    #X_pred = pd.DataFrame(columns=["key", "pickup_datetime", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count"], data=[answer])
    #loaded_model = joblib.load('model.joblib')
    
    return {'test de predict': 'ça roule'}

#loaded_model.predict(X_pred)[0]}
'''