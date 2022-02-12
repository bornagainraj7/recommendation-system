from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pandas as pd
import pickle
import nltk
import model
import os


# nltk_data_folder = './data/nltk/'

# if not os.path.exists(nltk_data_folder):
#     os.makedirs(nltk_data_folder)

# nltk.download('punkt', download_dir = nltk_data_folder)

app = Flask(__name__)


@app.route('/')
def index():
    # fetching all username list
    allUsername = model.recommendation_model.index.tolist()
    return render_template('index.html', usernameList = allUsername)


@app.route('/recommend', methods = ['POST'])
def recommend():
    username = str(request.form.get('username'))
    
    print('username ', username)
    if not username:
        return redirect(url_for('index'))

    productNameList, posSentimentRateList = model.doRecommendations(username)

    if  posSentimentRateList == None or type(productNameList) == 'str':
        allUsername = model.recommendation_model.index.tolist()
        return render_template('index.html', usernameList = allUsername, error = productNameList)

    productList = zip(productNameList, posSentimentRateList)

    return render_template('recommendations.html', username = username, productList = productList)


if __name__ == '__main__':
    print('===============Flask App Started=============')
    print('Sentiment Based Product Recommendation System')
    app.run(debug = True)
