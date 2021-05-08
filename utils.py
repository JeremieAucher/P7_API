# -*- coding: utf-8 -*-
"""
Created on Thu May  6 19:22:26 2021

@author: waldu
"""

# import os
import pickle
import base64
import numpy as np
import urllib.request


def modelPredict(model, data, threshold):
    '''
        Retourne la prédiction du modèle: 0 ou 1 en fonction du seuil
        ainsi que la valeur exact de probabilité donné par le modèle.
    '''
    
    pP = model.predict_proba(data)[:,0].item()
    pE = int(np.where(pP<threshold,0,1))
    
    return convToB64(
        dict(
            predProba = pP,
            predExact = pE
            )
        )

def loadModelLightGBM(formatFile='b64'):
    model = pickle.load(open('/pickle/model.pkl', 'rb'))
    # model = pickle.load(open(os.getcwd()+'/pickle/model.pkl', 'rb'))
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # model = pickle.load(open(dir_path+'/pickle/model.pkl', 'rb'))
    # model = pickle.load(urllib.request.urlopen("https://p7-api-public.s3.eu-west-3.amazonaws.com/model.pkl"))
    
    if formatFile == 'pkl':
        return model
    elif formatFile == 'b64':
        return convToB64(model)
    else:
        return model.class_weight
        
def convToB64(data):
    return base64.b64encode(pickle.dumps(data)).decode('utf-8')

def restoreFromB64Str(data_b64_str):
    return pickle.loads(base64.b64decode(data_b64_str.encode()))
