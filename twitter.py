import requests
import base64
from config import CONSUMER_KEY, CONSUMER_SECRET, BASE_URL, NUM_TWEETS
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import pickle

def setup_api():
    key_secret = '{}:{}'.format(CONSUMER_KEY, CONSUMER_SECRET).encode('ascii')
    b64_encoded_key = base64.b64encode(key_secret)
    b64_encoded_key = b64_encoded_key.decode('ascii')

    auth_url = '{}oauth2/token'.format(BASE_URL)

    auth_headers = {
        'Authorization': 'Basic {}'.format(b64_encoded_key),
        'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'
    }

    auth_data = {
        'grant_type': 'client_credentials'
    }

    auth_resp = requests.post(auth_url, headers=auth_headers, data=auth_data)

    access_token = auth_resp.json()['access_token']
    return access_token

def get_tweets(access_token, topic):
    search_headers = {
        'Authorization': 'Bearer {}'.format(access_token)
    }

    search_params = {
        'q': topic,
        'result_type': 'recent',
        'count': NUM_TWEETS
    }

    search_url = '{}1.1/search/tweets.json'.format(BASE_URL)

    search_resp = requests.get(search_url, headers=search_headers, params=search_params)
    tweet_data = search_resp.json()

    return [t['text'] for t in tweet_data['statuses']]


def prepare_tweet(tweet):
    tokenizer = pickle.load(open("tokenizer.pkl", 'rb'))
    return sequence.pad_sequences(tokenizer.texts_to_sequences(tweet), maxlen=100, value=0)