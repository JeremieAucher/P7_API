# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 23:00:45 2021

@author: waldu
"""

# -*- coding: utf-8 -*-

from flask import Flask, request
import numpy as np
# import jsonify
from flask_jsonpify import jsonify
import pickle
import os

### Initialisation ###
app = Flask(__name__)
model = pickle.load(open(os.getcwd()+'/pickle/model.pkl', 'rb'))
threshold = 0.50
#####################

### Déclaration des fonctions - Start ###

# def loadModel():
#     return pickle.load(open(os.getcwd()+'\\pickle\\model.pkl', 'rb'))

# def loadModel():
#     return pickle.load(open(os.getcwd()+'/pickle/model.pkl', 'rb'))

# def getTheIDX(data,value,columnName='SK_ID_CURR'):
#     '''
#     Retourne l'index correspondant à la 1ère valeur contenue dans value
#     contenue dans la colonne columnName du Dataframe data.
#     ''' 
#     return data[data[columnName] == value].index[0]

def modelPredict(data):
    '''
        Retourne la prédiction du modèle: 0 ou 1 en fonction du seuil
        ainsi que la valeur exact de probabilité donné par le modèle.
    '''
    # idx = getTheIDX(data=data,columnName=loanColumn,value=loanNumber)
    # resultModel = model.predict_proba(data[data.index == idx])[:,1]
    
    predExact = model.predict_proba(data)[:,1]
    predProba = np.where(predExact<threshold,0,1)[0]
    
    return jsonify({
            'predExact':predExact,
            'predProba':predProba
            
        })

### Déclaration des fonctions - End ###

### app.route - Start ###
@app.route('/lightgbm/')
def lightgbm():
    # A corriger: il faut envoyer juste une ligne de donné, et non le DF en entier.
    # La selection de la ligne de donné à envoyer doit être fait avant l'entrée dans la fonction modelPredict.
    
    # Réccupération des données
    data = request.args.get('data')
    
    return modelPredict(data)
    
    # predExact, predProba = modelPredict(data)
    # return jsonify({
    #         'predExact':predExact,
    #         'predProba':predProba
            
    #     })

@app.route('/test/')
def test():
    # if doesn't exist, returns None
    language = request.args.get('language')
    return '''<h1>The language value is: {}</h1>'''.format(language)

@app.route('/test2/')
def test2():
    # if doesn't exist, returns None
    language = request.args.get('language')
    return jsonify(language)

@app.route('/')
def helloworld():
    return '''<h1>Hello World!</h1>'''

### app.route - End ###




# app.run()


# from flask import Flask, render_template, jsonify
# import json
# import requests

# app = Flask(__name__)

# # METEO_API_KEY = "c30c785207dc7f397b5c036ba5fc70xx"

# # if METEO_API_KEY is None:
# #     # URL de test :
# #     METEO_API_URL = "https://samples.openweathermap.org/data/2.5/forecast?lat=0&lon=0&appid=xxx"
# # else: 
# #     # URL avec clé :
# #     METEO_API_URL = "https://api.openweathermap.org/data/2.5/forecast?lat=48.883587&lon=2.333779&appid=" + METEO_API_KEY

# @app.route("/")
# def hello():
#     return "Hello World!"

# @app.route('/dashboard/')
# def dashboard():
#     return render_template("dashboard.html")

if __name__ == "__main__":
    # app.run(debug=True)
    app.run()