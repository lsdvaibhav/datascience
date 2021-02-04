from flask import Flask, render_template, make_response, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import webbrowser
import bs4 as bs 
import urllib.request
import requests
import sys
import os
import glob
import re

import numpy as np
import pandas as pd

import sweetviz as sv
from pandas_profiling import ProfileReport
from pandas_profiling.utils.cache import cache_file
import json

from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer


app = Flask(__name__)



# Movie recommendation section 

# load the nlp model and tfidf vectorizer from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl','rb'))

def create_similarity():
    data = pd.read_csv('datasets/main_data.csv')
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return data,similarity

def rcmd(m):
    m = m.lower()
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    if m not in data['movie_title'].unique():
        return('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
        i = data.loc[data['movie_title']==m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:11] # excluding first item since it is the requested movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l
    
# converting list of string to list (eg. "["abc","def"]" to ["abc","def"])
def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list

def get_suggestions():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())


#root app part
@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route("/movie" ,methods=['POST','GET'])
def home():
    suggestions = get_suggestions()
    return render_template('home.html',suggestions=suggestions)

@app.route("/similarity",methods=["POST"])
def similarity():
    movie = request.form['name']
    rc = rcmd(movie)
    if type(rc)==type('string'):
        return rc
    else:
        m_str="---".join(rc)
        return m_str

@app.route("/recommend",methods=["POST"])
def recommend():
    # getting data from AJAX request
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    # get movie suggestions for auto complete
    suggestions = get_suggestions()

    # call the convert_to_list function for every string that needs to be converted to list
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)
    
    # convert string to list (eg. "[1,2,3]" to [1,2,3])
    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[","")
    cast_ids[-1] = cast_ids[-1].replace("]","")
    
    # rendering the string to python string
    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"','\"')
    
    # combining multiple lists as a dictionary which can be passed to the html file so that it can be processed easily and the order of information will be preserved
    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
    
    casts = {cast_names[i]:[cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}

    cast_details = {cast_names[i]:[cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

    # web scraping to get user reviews from IMDB site
    sauce = urllib.request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).read()
    soup = bs.BeautifulSoup(sauce,'lxml')
    soup_result = soup.find_all("div",{"class":"text show-more__control"})

    reviews_list = [] # list of reviews
    reviews_status = [] # list of comments (good or bad)
    for reviews in soup_result:
        if reviews.string:
            reviews_list.append(reviews.string)
            # passing the review to our model
            movie_review_list = np.array([reviews.string])
            movie_vector = vectorizer.transform(movie_review_list)
            pred = clf.predict(movie_vector)
            reviews_status.append('Good' if pred else 'Bad')

    # combining reviews and comments into a dictionary
    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}     

    # passing all the data to the html file
    return render_template('recommend.html',title=title,poster=poster,overview=overview,vote_average=vote_average,
        vote_count=vote_count,release_date=release_date,runtime=runtime,status=status,genres=genres,
        movie_cards=movie_cards,reviews=movie_reviews,casts=casts,cast_details=cast_details)




    


@app.route('/credit',methods=['POST','GET'])
def credit():
    if request.method== "POST":
    
        model=pickle.load(open('credit_card_fraud.pkl','rb'))
        
        features =[v for v in request.form.values()]
        print(features)
        
        final=[np.array(features)]
        pred = model.predict(final)
           
        pred = int(pred[0])
        if pred == 0:
            message= "transaction does not fraud "
            alert = "success"
        else:
            message=" Transaction found fraud"
            alert= "danger"
        res = make_response(jsonify({"message": message , "alert" : alert }), 200)

        return res
    return render_template("index.html")
    
@app.route("/eda", methods=["GET", "POST"])
def upload_csv():
    if request.method == "POST":

        file = request.files["file"]
        
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(file.filename))
        file.save(file_path)
        
        #processing and generating eda report in uploads folder
        df = pd.read_csv(file_path)
        filename_ ="uploads/"+file.filename+".html"
        profile = ProfileReport(df, title= file.filename, explorative=True)
        profile.to_file(filename_)
        sweetviz_eda=sv.analyze(df)
        
        res = make_response(jsonify({"message": "File "+file.filename+" uploaded successfully and report is Ready. "}), 200)
        #showing both reports in new tabs
        webbrowser.open_new_tab(file_path+".html")
        sweetviz_eda.show_html()
        return res 
      
    return render_template("inex.html")




@app.route("/pnemonia", methods=["GET", "POST"])
def upload_img():
    if request.method == "POST":

        file = request.files["file"]
        
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(file.filename))
        file.save(file_path)
        
        
        res = make_response(jsonify({"message": "Pneumonia Detected "}), 200)

        return res 
      
    return render_template("index.html")

    
    
@app.route('/ff_process', methods=['POST','GET'])
def ff():
    if request.method == "POST":

        file = request.files["file"]
        oxy = int(request.form["oxy"])
        temp = int(request.form["temp"])
        hum = int(request.form["hum"])
        
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(file.filename))
        file.save(file_path)
        
        
        model=pickle.load(open('ff_model.pkl','rb'))
        inputt=[oxy ,temp ,hum ]
        final=[np.array(inputt)]

        b = model.predict_proba(final)
        if b[0][0]>0.5:
           message = "Forest is Safe take action vicely. Accuracy : {} ".format(int(b[0][0]*100))
           alert ="success"
        else :
            message =" Forest is in Danger take action immedietly. Accuracy :{} ".format(int(b[0][1]*100))
            alert="danger"
        
        res = make_response(jsonify({"message": message ,"alert": alert }), 200)

        return res 
      
    return render_template("index.html")
      
      
      

    
@app.route('/spam', methods=['POST'])
def spam():
    if request.method == 'POST':
        df= pd.read_csv("YoutubeSpamMergedData.csv")
        df_data = df[["CONTENT","CLASS"]]
        # Features and Labels
        df_x = df_data['CONTENT']
        df_y = df_data.CLASS
        # Extract Feature With CountVectorizer
        corpus = df_x
        cv = CountVectorizer()
        X = cv.fit_transform(corpus) # Fit the Data
        
        X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)
        #Naive Bayes Classifier
        
        clf = MultinomialNB()
        clf.fit(X_train,y_train)
        clf.score(X_test,y_test)
    
   
        comment = request.form['comment']
      
        data = [comment]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        print(my_prediction)
        if my_prediction == 1:
            message ="  Its a Spam."
            alert = "warning"
        else:
            message ="  Its a ham."
            alert = "success"
            
            
        res = make_response(jsonify({"message": message , "alert" :alert}), 200)

        return res 

    return render_template("index.html")  
    
    
#chatbot section
    
english_bot = ChatBot("Chatterbot", storage_adapter="chatterbot.storage.SQLStorageAdapter")
trainer = ChatterBotCorpusTrainer(english_bot)
trainer.train("chatterbot.corpus.english")



@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(english_bot.get_response(userText))
    
    
if __name__ == '__main__':
    app.run(debug=True)
