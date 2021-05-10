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
import os

### Initialisation ###
app = Flask(__name__)
mo = utils.loadModelLightGBM(formatFile='pkl') # Named mo because a function named model already exists
cols = utils.loadColumnsOfModel()
th = 0.50 # Named th because a function named threshold already exists
txtB64Global = ''
dictTxtB64Split = {}
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
    global dictTxtB64Split
    # init txB64Global
    try:
        txtB64Global = ''
        dictTxtB64Split = {}
        print('Hello world!', file=sys.stderr)
        print(f'Keys de dictTxtB64Split={dictTxtB64Split.keys()}', file=sys.stderr)
        return '1'
    except:
        return '0'

@app.route('/merge/',methods=['POST'])
def splitN():
    global txtB64Global
    global dictTxtB64Split
    print(f'Merge - numSplit={request.values.get("numSplit")}', file=sys.stderr)
    print(f'Avant traitement - Keys de dictTxtB64Split={dictTxtB64Split.keys()}', file=sys.stderr)
    # Recept Split n 
    # txtB64Global += request.values.get('txtSplit') # Pas adapté sur Heroku
    n = request.values.get("numSplit")
    print(f'numSplit = {n} --- Type numSplit = {type(n)} --- int(numSplit) = {int(n)} --- Type int(numSplit) = {type(int(n))}')
    dictTxtB64Split[int(request.values.get("numSplit"))] = request.values.get('txtSplit')
    
    print(f'Après traitement - Keys de dictTxtB64Split={dictTxtB64Split.keys()}', file=sys.stderr)
    return '1'

@app.route('/endSplit/',methods=['POST'])
def endSplit():
    global txtB64Global
    global dictTxtB64Split
    
    print('endSplit', file=sys.stderr)
    print(f'Keys de dictTxtB64Split: {dictTxtB64Split.keys()}')
    
    # Restore data
    for i in range(5):
        txtB64Global += dictTxtB64Split[i]
    
    print(f'Len de txtB64Global={len(txtB64Global)}', file=sys.stderr)
    
    # Decode Data
    dataOneCustomer = utils.restoreFromB64Str(txtB64Global)
    
    # Creation de dfOneCustomer
    dfOneCustomer = pd.DataFrame(data=dataOneCustomer, columns=cols)
    
    # Intérrogation du model et retour des résultats
    return utils.modelPredict(mo,dfOneCustomer,th)

@app.route('/writeFile/',methods=['POST'])
def writeFile():
    fp = open("testWrite.txt", 'w')
    fp.write('Ceci est un test')
    fp.close()
    return '1'

@app.route('/writeFile2/',methods=['POST'])
def writeFile2():
    
    # créer le dossier s'il n'existe pas
    if not os.path.exists('TEST'):
    os.makedirs('TEST')
    
    fp = open("/TEST/testWrite.txt", 'w')
    fp.write('Ceci est un second test')
    fp.close()
    return '1'

@app.route('/readFile/',methods=['POST'])
def readFile():
    fp = open("testWrite.txt", 'r')
    contents = fp.read()
    print(f'Contenu de testWrite.txt: {contents}', file=sys.stderr)
    return contents

@app.route('/readFile2/',methods=['POST'])
def readFile2():
    fp = open("/TEST/testWrite.txt", 'r')
    contents = fp.read()
    print(f'Contenu de testWrite.txt: {contents}', file=sys.stderr)
    return contents

@app.route('/deleteDir/',methods=['POST'])
def deleteDir():
    os.rmdir('TEST')
    return '1'


### app.route - End ###

if __name__ == "__main__":
    app.run()