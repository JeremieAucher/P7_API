# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 23:00:45 2021

@author: waldu
"""

# -*- coding: utf-8 -*-

from flask import Flask, request
import utils

### Initialisation ###
app = Flask(__name__)
mo = utils.loadModelLightGBM(formatFile='pkl') # Named mo because a function named model already exists
th = 0.50 # Named th because a function named threshold already exists
#####################

### app.route - Start ###
@app.route('/lightgbm/',methods=['POST'])
def lightgbm():
    return utils.modelPredict(mo,utils.restoreFromB64Str(request.args.get('data_b64_str')),th)

@app.route('/model/',methods=['POST'])
def model():
    return utils.loadModelLightGBM(formatFile='b64')

@app.route('/test/')
def test():
    return f'''{utils.loadModelLightGBM(formatFile='xxx')}'''

@app.route('/test2/')
def test2():
    return '''The scikit-learn version is {sklearn.__version__}'''

@app.route('/threshold/',methods=['POST'])
def threshold():
    return utils.convToB64(th)

@app.route('/')
def helloworld():
    return '''<h1>Bienvenue sur la partie API du P7 DataScientist d'OpenClassrooms'</h1>'''

### app.route - End ###

if __name__ == "__main__":
    app.run()