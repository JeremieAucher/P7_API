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
    # return request.args.get('data_b64_str')

@app.route('/abc/',methods=['POST'])
def abc():
    return utils.modelPredict(mo,utils.restoreFromB64Str(request.args.get('abc_b64_str')),th)

@app.route('/wxc/',methods=['POST'])
def wxc():
    return request.form.get('wxc_b64_str')

@app.route('/qsd/',methods=['POST'])
def qsd():
    return utils.modelPredict(mo,utils.restoreFromB64Str(request.form.get('qsd_b64_str')),th)

@app.route('/aaa/',methods=['GET'])
def aaa():
    return request.args.get('XXX')

@app.route('/bbb/',methods=['POST'])
def bbb():
    return request.args("XXX")

@app.route('/ccc/',methods=['POST'])
def ccc():
    return request.values.get('XXX')

@app.route('/ddd/',methods=['POST'])
def ddd():
    return utils.modelPredict(mo,utils.restoreFromB64Str(request.values.get('data_b64_str')),th)

@app.route('/model/',methods=['POST'])
def model():
    return utils.loadModelLightGBM(formatFile='b64')



@app.route('/test/')
def test():
    return f'''{utils.loadModelLightGBM(formatFile='xxx')}'''

@app.route('/threshold/',methods=['POST'])
def threshold():
    return utils.convToB64(th)

@app.route('/')
def helloworld():
    return '''<h1>Bienvenue sur la partie API du P7 DataScientist d'OpenClassrooms'</h1>'''

### app.route - End ###

if __name__ == "__main__":
    app.run()