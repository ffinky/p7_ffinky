# Data manipulation
import numpy as np

# Création API
from fastapi import FastAPI
from pydantic import BaseModel

# Import du modèle sauvegardé de classification en tant qu'objet (désérialisation)
import pickle
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

# Chargement du modèle de classification en fonction du choix utilisateur
def fct_load_classifier(choice_model):
    if choice_model == 'Reg Logistique':
        classifier_file = 'log_model_final.p'
    else:
        classifier_file = 'lgbm_model_final.p'
    #classifier = pickle.load(open(classifier_file, 'rb'))
    with open(classifier_file, 'rb') as f:
        classifier = pickle.load(f)
    return classifier

# Extraction des features reçues de l'appli (JSON) et conversion au format Numpy X depuis 1 dictionnaire 
def fct_extract_X_from_dict(dict_feats):
    # Liste de (key, val)
    feats = list(dict_feats.items())
    # Format Numpy 
    X = np.array(feats)
    # Colonne 1 contenant les valeurs 
    X = X[:,1] 
    # Permutation : redimensionnement des n lignes en (1 ligne, n features)
    X = X.reshape([1, X.shape[0]])
    # Retourne les features X au format Numpy array
    return X

# Calcul du score de prédiction d'octroi de crédit
def fct_score_calculate(classifier_name, dict_feats):
    # Chargement du classifier choisi par l'utilisateur
    classifier = fct_load_classifier(classifier_name)
    print('Load classifier OK')
    
    # Extraction des features au format Numpy
    X = fct_extract_X_from_dict(dict_feats)

    # Prédiction du score de probabilité d'octroi de crédit 
    print('Calcul du score...')
    y_score_pred = classifier.predict_proba(X)[:, 1]
    print('Calcul réussi')
    
    # Retourne le score prédit entre 0 et 1
    return np.round(y_score_pred[0], 3)

# Arguments de l'URL de l'API
class user_input(BaseModel):
    classifier_name : str
    features : dict

# On instancie le serveur web API (back-end)
app = FastAPI()

# URL route
@app.get("/")
async def root():
    return {'message': 'Hello, API is working'}

# URL de l'API de calcul du score
@app.post('/score_calculate')
def operate(input:user_input):
    result =  fct_score_calculate(input.classifier_name, 
                                  input.features)
    return result