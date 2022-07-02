from flask import Flask, render_template, url_for, request, redirect
from joblib import load
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
import spacy
from tqdm.auto import tqdm

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def project_NLP():
    request_type = request.method

    if request_type == 'GET':
        return render_template('index.html')
    else:
        entered = request.form['text-to-predict']

        text_v = np.zeros((1, 11, 300))
        nlp = spacy.load("en_core_web_md")
        for i, token in enumerate(nlp(entered)):
            if i >= 11:
                break
            text_v[0, i] = token.vector
        # predict the input
        #model = load("model.joblib")
        model = tf.keras.models.load_model('saved_model/my_model')
        prediction = model.predict(text_v)
        encoder = LabelEncoder()
        encoder.classes_ = np.load('classes.npy',allow_pickle=True)
        label = encoder.classes_[np.argmax(prediction)]

        return render_template('result.html', label=label.strip(''))





if __name__ == "__main__":
    app.run(host = '0.0.0.0')
