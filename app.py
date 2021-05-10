# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 23:00:45 2021

@author: waldu
"""

# -*- coding: utf-8 -*-

from flask import Flask, request
import utils
import pandas as pd
import sys

### Initialisation ###
app = Flask(__name__)
mo = utils.loadModelLightGBM(formatFile='pkl') # Named mo because a function named model already exists
cols = utils.loadColumnsOfModel()
th = 0.50 # Named th because a function named threshold already exists
txtB64Global = ''
#####################

### app.route - Start ###
@app.route('/lightgbm/',methods=['POST'])
def lightgbm():
    return utils.modelPredict(mo,utils.restoreFromB64Str(request.args.get('data_b64_str')),th)
    # return request.args.get('data_b64_str')

@app.route('/threshold/',methods=['POST'])
def threshold():
    return utils.convToB64(th)

@app.route('/')
def helloworld():
    return '''<h1>Bienvenue sur la partie API du P7 DataScientist d'OpenClassrooms'</h1>'''

@app.route('/ccc/',methods=['POST'])
def ccc():
    return request.values.get('XXX')


@app.route('/initSplit/',methods=['POST'])
def startSplit():
    global txtB64Global
    # init txB64Global
    try:
        txtB64Global = ''
        print('Hello world!', file=sys.stderr)
        print(f'Len de txtB64Global={len(txtB64Global)}', file=sys.stderr)
        return '1'
    except:
    	return '0'

@app.route('/merge/',methods=['POST'])
def splitN():
    global txtB64Global
    print(f'Merge - numSplit={request.values.get("txtSplit")}', file=sys.stderr)
    print(f'Avant traitement - Len de txtB64Global={len(txtB64Global)}', file=sys.stderr)
    # Recept Split n 
    txtB64Global += request.values.get('txtSplit')
    print(f'Après traitement - Len de txtB64Global={len(txtB64Global)}', file=sys.stderr)
    return '1'

@app.route('/endSplit/',methods=['POST'])
def endSplit():
    global txtB64Global
    print('endSplit', file=sys.stderr)
    print(f'Len de txtB64Global={len(txtB64Global)}', file=sys.stderr)
    
    # Decode and restor the Data
    dataOneCustomer = utils.restoreFromB64Str(txtB64Global)
    
    # Creation de dfOneCustomer
    dfOneCustomer = pd.DataFrame(data=dataOneCustomer, columns=cols)
    
    # Intérrogation du model et retour des résultats
    return utils.modelPredict(mo,dfOneCustomer,th)

### app.route - End ###

if __name__ == "__main__":
    app.run()