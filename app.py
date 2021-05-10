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
import pickle
import shutil

### Initialisation ###
app = Flask(__name__)
mo = utils.loadModelLightGBM(formatFile='pkl') # Named mo because a function named model already exists
cols = utils.loadColumnsOfModel()
th = 0.50 # Named th because a function named threshold already exists
MYDIR = os.path.dirname(__file__)
# formatOsSlash = '\\'
formatOsSlash = '/'
tmpDirName = 'tmpSplit'
tmpDir = formatOsSlash+tmpDirName+formatOsSlash
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
def initSplit():
    global MYDIR
    global tmpDirName

    print('initSplit', file=sys.stderr)

    # On Initialise le dossier de destination
    # On l'efface s'il existe, avec tout ce qu'il contient
    shutil.rmtree(tmpDirName, ignore_errors=True)
    # On crée le dossier temporaire
    if not os.path.exists(tmpDirName):
        os.makedirs(tmpDirName)
    return '1'

@app.route('/merge/',methods=['POST'])
def splitN():
    global MYDIR
    global tmpDir
    pathFile = MYDIR+tmpDir+request.values.get("numSplit")+'.pkl'
    strToSave = request.values.get('txtSplit')
    
    print(f'Merge - numSplit={request.values.get("numSplit")}', file=sys.stderr)

    # On enregistre le contenu reç dans un fichier pickle
    # Le fichier pickle porte le nom du numero de split
    pickle.dump(strToSave, open(pathFile, 'wb'))
    return '1'

@app.route('/endSplit/',methods=['POST'])
def endSplit():
    global MYDIR
    global tmpDir
    global tmpDirName
    txtB64Global = ''

    print('endSplit', file=sys.stderr)
    
    # Restore data
    for i in range(5):
        pathFile = MYDIR+tmpDir+str(i)+'.pkl'
        # On ouvre le fichier pickle et on joint son contenu au txt global
        txtB64Global += pickle.load(open(pathFile, 'rb'))
    

    # On restore les données

    print(f'Len de txtB64Global={len(txtB64Global)}', file=sys.stderr)
    
    # Decode Data
    dataOneCustomer = utils.restoreFromB64Str(txtB64Global)
    
    # Creation de dfOneCustomer
    dfOneCustomer = pd.DataFrame(data=dataOneCustomer, columns=cols)
    
    # Par sécurité on efface le dossier temporaire
    shutil.rmtree(tmpDirName, ignore_errors=True)
    
    # Intérrogation du model et retour des résultats
    return utils.modelPredict(mo,dfOneCustomer,th)


### app.route - End ###

if __name__ == "__main__":
    app.run()













# @app.route('/initSplit/',methods=['POST'])
# def startSplit():
#     global txtB64Global
#     global dictTxtB64Split
#     # init txB64Global
#     try:
#         txtB64Global = ''
#         dictTxtB64Split = {}
#         print('Hello world!', file=sys.stderr)
#         print(f'Keys de dictTxtB64Split={dictTxtB64Split.keys()}', file=sys.stderr)
#         return '1'
#     except:
#         return '0'

# @app.route('/merge/',methods=['POST'])
# def splitN():
#     global txtB64Global
#     global dictTxtB64Split
#     print(f'Merge - numSplit={request.values.get("numSplit")}', file=sys.stderr)
#     print(f'Avant traitement - Keys de dictTxtB64Split={dictTxtB64Split.keys()}', file=sys.stderr)
#     # Recept Split n 
#     # txtB64Global += request.values.get('txtSplit') # Pas adapté sur Heroku
#     n = request.values.get("numSplit")
#     print(f'numSplit = {n} --- Type numSplit = {type(n)} --- int(numSplit) = {int(n)} --- Type int(numSplit) = {type(int(n))}')
#     dictTxtB64Split[int(request.values.get("numSplit"))] = request.values.get('txtSplit')
    
#     print(f'Après traitement - Keys de dictTxtB64Split={dictTxtB64Split.keys()}', file=sys.stderr)
#     return '1'

# @app.route('/endSplit/',methods=['POST'])
# def endSplit():
#     global txtB64Global
#     global dictTxtB64Split
    
#     print('endSplit', file=sys.stderr)
#     print(f'Keys de dictTxtB64Split: {dictTxtB64Split.keys()}')
    
#     # Restore data
#     for i in range(5):
#         txtB64Global += dictTxtB64Split[i]
    
#     print(f'Len de txtB64Global={len(txtB64Global)}', file=sys.stderr)
    
#     # Decode Data
#     dataOneCustomer = utils.restoreFromB64Str(txtB64Global)
    
#     # Creation de dfOneCustomer
#     dfOneCustomer = pd.DataFrame(data=dataOneCustomer, columns=cols)
    
#     # Intérrogation du model et retour des résultats
#     return utils.modelPredict(mo,dfOneCustomer,th)


# @app.route('/writePkl/',methods=['POST'])
# def writePkl():
#     global MYDIR
#     global tmpDir
#     pathFile = MYDIR+tmpDir+request.values.get('fileName')+'.pkl'
    
#     # Création du dossier
#     if not os.path.exists(tmpDir.replace('\\','')):
#         os.makedirs(tmpDir.replace('\\',''))
    
#     print(f"Dossier actuel: {os.getcwd()}", file=sys.stderr)
#     print(f'MYDIR: {MYDIR}', file=sys.stderr)
    
#     pdTest = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
#     pickle.dump(pdTest, open(pathFile, 'wb'))
    
#     return '1'
    
# @app.route('/sendPkl/',methods=['POST'])
# def sendPkl():
#     global MYDIR
#     global tmpDir
#     pathFile = MYDIR+tmpDir+request.values.get('fileName')+'.pkl'
    
#     pdTest = pickle.load(open(pathFile, 'rb'))
    
#     return utils.convToB64(pdTest)
    
    

    
# @app.route('/writeFile/',methods=['POST'])
# def writeFile():
#     global MYDIR
#     global tmpDir
#     # MYDIR = os.path.dirname(__file__)
#     print(f"Dossier actuel: {os.getcwd()}", file=sys.stderr)
#     print(f'MYDIR: {MYDIR}', file=sys.stderr)
    
#     # créer le dossier s'il n'existe pas
#     # print(f"Dossier {tmpDir} existe?: {os.path.exists(tmpDir.replace('\\',''))}", file=sys.stderr)
#     if not os.path.exists(tmpDir.replace('\\','')):
#         os.makedirs(tmpDir.replace('\\',''))
    
    
#     # fp = open("/TEST/testWrite.txt", 'w')
#     fp = open(MYDIR+tmpDir+'testWrite.txt', 'w')
#     fp.write('Ceci est un second test')
#     fp.close()
#     return '1'


# @app.route('/readFile/',methods=['POST'])
# def readFile():
#     global MYDIR
#     global tmpDir
#     # MYDIR = os.path.dirname(__file__)
#     print(f'MYDIR: {MYDIR}', file=sys.stderr)
#     fp = open(MYDIR+tmpDir+'testWrite.txt', 'r')
#     contents = fp.read()
#     print(f'Contenu de testWrite.txt: {contents}', file=sys.stderr)
#     return contents

# @app.route('/deleteDir/',methods=['POST'])
# def deleteDir():
#     global tmpDir
#     import shutil
#     shutil.rmtree(tmpDir.replace('\\', ''), ignore_errors=True)
#     # os.rmdir('TEST')
#     return '1'


