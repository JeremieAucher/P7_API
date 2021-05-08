# -*- coding: utf-8 -*-
"""
Created on Thu May  6 19:22:26 2021

@author: waldu
"""

import os
import pickle
import base64


def modelPredict(model, data):
    '''
        Retourne la prédiction du modèle: 0 ou 1 en fonction du seuil
        ainsi que la valeur exact de probabilité donné par le modèle.
    '''
    
    return convToB64(
        dict(
            predProba = model.predict_proba(data)[:,0].item(),
            predExact = int(np.where(predProba<threshold,0,1))
            )
        )
    
    # return {
    #     'predProba':predProba,
    #     'predExact':predExact
    #     }

def loadModelLightGBM(formatFile='b64'):
    model = pickle.load(open(os.getcwd()+'/pickle/model.pkl', 'rb'))
    
    if formatFile == 'pkl':
        return model
    elif formatFile == 'b64':
        return convToB64(model)
    else:
        return 0
        
def convToB64(data):
    return base64.b64encode(pickle.dumps(data)).decode('utf-8')

def restoreFromB64Str(data_b64_str):
    return pickle.loads(base64.b64decode(data_b64_str.encode()))
