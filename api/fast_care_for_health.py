from pyexpat import model
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
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
    select_meds='tous',
    sort_how='calculated',
    select_radius=15,
    breakeven_rate=None,
    neighbors_weight=0.1,
    min_neighbors=2,
    code_region=None,
    ):

    params=dict(
        selection_medecins=str(select_meds), 
        sortby=str(sort_how),
        radius=int(select_radius),
        moy_region=str(breakeven_rate),
        poids_des_voisins=float(neighbors_weight),
        nb_voisins_minimum=int(min_neighbors),
        region=code_region
        )

    if code_region == None or code_region == 'None':
        pass
    else:
        code_region=int(code_region)
    #df = pd.read_csv('data/df_api_test.csv', delimiter=',', dtype={'code_insee':'str'}, converters={"neighbors": lambda x: ast.literal_eval(x)}).reindex()
    df = pd.read_csv('data/df_api_france_9.csv', delimiter=',', dtype={'code_insee':'str'}, converters={"neighbors": lambda x: ast.literal_eval(x)}).reindex()
    if code_region:
        df = df[df["code_regions"].isin([code_region])].copy()
    
    model = model_algo.Medical_ReDispatch()
    model.set_params(**params)
    model.fit(df)
    y_pred = model.predict(df)
    y_dict = y_pred.set_index('code_insee').to_dict()
    
    scr = model.score(df)
    return {
        'Ancien_taux_moyenne_communes': scr['ancien_taux'],
        'Nouveau_taux_moyenne_communes': scr['nouveau_taux'],
        'Evolution du taux': scr['delta_taux'],
        'Ancien nombre de communes déficitaires': scr['ancien_communes_déficitaires'],
        "Communes n'étant plus déficitaires": scr['delta_communes'],
        'Distance_moyenne_parcourue': scr['distance_moyenne'],
        'Distance_totale_parcourue': f"{scr['distance_moyenne']*scr['médecins_déplacés']:.2f}",
        'Nombre total de médecins': scr['total_medecins'],
        'Médecins déplacés': scr['médecins_déplacés'],
        'Taux initial pondéré': scr['initial_weighted_rate'],
        'Taux final pondéré': scr['final_weighted_rate'],
        'Data':json.dumps(y_dict)
    }