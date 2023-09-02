from flask import Flask, request
from flask_cors import CORS
import numpy as np
import nltk
nltk.download('punkt')
import tflearn
import tensorflow as tf
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import json
import random
import pickle

# import os
# os.system('python -m pip install --upgrade pip')

app = Flask(__name__)
CORS(app)

app.config["SECRET_KEY"] = "faksjdlfkajsdfrr"   

data = {}

with open('./intents.json') as f:
    data = json.load(f)

print("Data read complete!")

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

print("Data sorted!")

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch = 1000, batch_size=8, show_metric=True)

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

@app.route('/', methods=["GET"])
def getHome():
    inp = request.get_json()

    inp = inp["data"]

    print(inp)

    results = model.predict([bag_of_words(inp, words)])
    results_index = np.argmax(results)
    tag = labels[results_index]

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
            response_list = nltk.sent_tokenize(str(responses[0]))
            return {"response": response_list}

    return {"response": "No response"}

if __name__ == "__main__":
    app.run()