# -*- coding: utf-8 -*-
import polyglot
from polyglot.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import flask
from flask import request, jsonify
from flask import json,Response
from flask_cors import CORS, cross_origin




from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = polytlog.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

class Responsee:   
    reponse="a"
    # The class "constructor" - It's actually an initializer 
    def __init__(self, reponse):
        self.reponse = reponse
        


app = flask.Flask(__name__)


app.config["DEBUG"] = False
app.config['JSON_AS_ASCII'] = False
app.config['CORS_HEADERS'] = "Content-Type"
cors = CORS(app, resources={r"/chat/*": {"origins": "*"}})



question=" "
res=""


@app.route('/chat/', methods=['GET'])
@cross_origin(origin='*',headers=['Content-Type'])

def chat():
    query_parameters = request.args
    question = query_parameters.get('question')
    res=chatbot_response(question)
    r=Responsee(res)
    x =res
   

    
    print(question) 
    return jsonify(x)

res=chatbot_response(question)



app.run()
