# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 23:00:45 2021

@author: waldu
"""

# -*- coding: utf-8 -*-

from flask import Flask, request
import utils
# from utils import restoreFromB64Str

### Initialisation ###
app = Flask(__name__)
# model = utils.loadModelLightGBM(formatFile='pkl')
threshold = 0.50
#####################

### app.route - Start ###
@app.route('/lightgbm/',methods=['POST'])
def lightgbm():
    # # Réccupération des données
    # data_b64_str = request.args.get('data_b64_str')
    # # Réencodage des données au format Pandas
    # # data = pickle.loads(base64.b64decode(data_b64_str.encode()))
    # data = restoreFromB64Str(request.args.get('data_b64_str'))
    # return utils.modelPredict(restoreFromB64Str(request.args.get('data_b64_str')))
    return utils.modelPredict(utils.restoreFromB64Str(request.args.get('data_b64_str')))

@app.route('/model/')
def model():
    return utils.loadModelLightGBM(formatFile='b64')
    
@app.route('/threshold/')
def threshold():
    return threshold

@app.route('/')
def helloworld():
    return '''<h1>Bienvenue sur la partie API du P7 DataScientist d'OpenClassrooms'</h1>'''

### app.route - End ###

if __name__ == "__main__":
    app.run()
    
# ImportError: cannot import name 'restoreFromB64Str' from 'utils' (E:\OneDrive\Documents\Formation_DataScientist_OpenClassroom\P7_\Dashboard-Streamlit\utils.py)