
from flask import Flask, request
import os
from keras.models import model_from_json
from twitter import setup_api, prepare_tweet, get_tweets
from main_page import page_html, make_result_page

app = Flask(__name__)

with open('model.json', 'r') as f:
    model = model_from_json(f.read())
    model.summary()
    model.load_weights('weights.h5')

@app.route('/')
def home():
    return page_html


@app.route('/get_pred', methods=["POST"])
def get_info():
    access_token = setup_api()
    topic = request.form['topic']
    tweets = get_tweets(access_token, topic)
    tot = 0
    most_positive = 0
    most_negative = 1
    pos_tweet = ""
    neg_tweet = ""
    for tweet in tweets:
        tweet_sequence = prepare_tweet(tweet)
        sent = model.predict(tweet_sequence)[0]
        tot += sent
        if sent > most_positive:
            most_positive = sent
            pos_tweet = tweet
        elif sent < most_negative:
            most_negative = sent
            neg_tweet = tweet
    average_sentiment = (tot/len(tweets))*100
    return make_result_page(topic, average_sentiment[0], pos_tweet, neg_tweet)


if __name__ == "main":
    port = os.environ.get('PORT')
    if port:
        app.run(host='0.0.0.0', port=int(port))
    else:
        app.run()
