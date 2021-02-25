## Project description

Sentiment analysis for tweets.

### Contributions:

Daniella: EDA, Sentiment analysis, presentation & Flask APP + Twitter API integration

Albert: LDA topic modeling & write up

Ruben: Hashtag classification & topic modeling

Rubens: EDA, Flask APP + Twitter API integration & write up

Eddie: Cleaning & preprocessing, sentiment analysis & presentation

## EDA

**Data Source 1: [Sentiment140 dataset with 1.6 million tweets](https://www.kaggle.com/kazanova/sentiment140)**

This data source is meant only to train our model to be able to distinguish between positive and negative tweets. Moving forward, we will use the trained model on data scrapped from twitter 

**Data Dictionary:** 

- **target**: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
- ids: The id of the tweet
- date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
- flag: The query (lyx). If there is no query, then this value is NO_QUERY
- user: the user that tweeted (robotickilldozr), text: the text of the tweet (Lyx is cool)
- **text**: the text of the tweet (Lyx is cool)

We are primarily interested in the text and target for now, but might later find it useful to use dates to make a time-dependent analysis.

This might be an issue as the topics covered in the tweets will be related to these dates. Our final model will be used in real-time data, and therefore we can either: **1.** Make the assumption that sentiment is time-independent (i.e - the way to determine if a tweet is positive/negative remained the same over the years) **2.** Use another dataset to have more varied inputs

- We found out that there were  1685 repeated tweets in the dataset, and removed these duplicates accordingly
- Through investigation we found that although the data-dictionary claims sentiment values from 0-4, the train data is split 50/50 between sentiment 0(negative) and 4(positive)
- We decided to convert those to 0/1 for simplicity
- For this initial exploration, we will only be testing a sentiment analysis model based on tweet and target, so we dropped the columns **"id", "flag" and "user"**
- Before preprocessing the text, we performed feature engineering to extract **hashtags**, which could indicate topics to investigate
- As a first preprocessing step, we made all words lowercase. Punctuation was not removed as in "twitter language" different punctuations can have contextual meaning. for that, we will later use a specific twitter tokenizer before removing punctuation
- Most common words in Positive/Negative tweets  are the same stop words

Conclusion: To obtain more interesting insights, we must analyze the most popular words for each category that are not in common, and also the most popular words once stop words are removed

Now we can see a more clear distinguishment between positive/negative words in the respective categories. 

- We then carried out the same step to investigate most popular **hashtags** in Positive/Negative tweets.
- We investigated the number of hashtags per tweet, but it seemed unrelated to the target

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4dc65be7-983d-4144-a6c5-bba87bb9e8ca/Screen_Shot_2021-01-24_at_10.31.23_AM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4dc65be7-983d-4144-a6c5-bba87bb9e8ca/Screen_Shot_2021-01-24_at_10.31.23_AM.png)

- Next preprocessing step was to remove the english **stop words**
- Once stop words were removed, we plotted a wordmap of the most popular words in positive and negative tweets

    **Negative tweet words**

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/289df5d3-e5ba-4b59-b203-04369ebe63fa/Screen_Shot_2021-01-24_at_10.47.53_AM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/289df5d3-e5ba-4b59-b203-04369ebe63fa/Screen_Shot_2021-01-24_at_10.47.53_AM.png)

     **Positive tweet words**

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c18aa347-d67e-4f5e-a5d2-f6218e10643a/Screen_Shot_2021-01-24_at_11.06.24_AM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c18aa347-d67e-4f5e-a5d2-f6218e10643a/Screen_Shot_2021-01-24_at_11.06.24_AM.png)

We did the same demonstration for hashtags

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/97232d7a-6c6a-4284-8d9a-9861312e8746/Screen_Shot_2021-01-24_at_10.56.00_AM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/97232d7a-6c6a-4284-8d9a-9861312e8746/Screen_Shot_2021-01-24_at_10.56.00_AM.png)

The hashtags are insightful as they often relate to **contemporary events**. We can see for exapmple the clear negative feeling towards the **iran-election** 

- Next, we **tokenized and lemmatized** words in tweets. We used the custom **tweet tokenizer** as it properly interprets special characters according to their meaning to tweets

## Modeling

### Modelling with Word2Vec & RNN

Bag of words:

We used Gensim’s Dictionary constructor to give each word in the tweet corpus a unique integer identifier.

Sentiment Analysis:

- We built an embedding layer using Word2Vec.
- Example of Word2Vec model working. Notice the interesting twitter-specific vocabulary such as h8:

```
[('dislike', 0.5086536407470703),
 ('fml', 0.4550246000289917),
 ('blah', 0.4027697443962097),
 ('swear', 0.40239256620407104),
 ('boring', 0.3992295265197754),
 ('urgh', 0.398406982421875),
 ('suck', 0.39819324016571045),
 ('killing', 0.396727591753006),
 ('hating', 0.3921506106853485),
 ('h8', 0.3761572241783142)]
```

 We then created a Tokenizer that we then saved to use in our Flask deployment. 

Then we built a NN architecture to predict sentiment.

- The first layer is a non-trainable embedding layer. The weights are the embedding matrix weights.
- Then, we used Dropout layers as regularization parameters (especially because we trained with a subset of the real data as it was taking too long, these dropouts were important to avoid overfitting)
- We then used Bidirectional LTSM layers with 128 units.
- Finally, a Dense layer with sigmoid activation for the binary classification

```python
nn_model = Sequential()
emb_layer = Embedding(vocab_size, 200, weights=[embedding_m], input_length=100, trainable=False)
nn_model.add(emb_layer)
nn_model.add(Dropout(rate=0.4))
nn_model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
nn_model.add(Dropout(rate=0.4))
nn_model.add(Bidirectional(LSTM(units=128, return_sequences=False)))
nn_model.add(Dense(units=1, activation='sigmoid'))
nn_model.add(Dense(1, activation='sigmoid'))
nn_model.compile(loss='binary_crossentropy',optimizer="adam",metrics=['accuracy'])

nn_model.summary()
callbacks = [EarlyStopping(monitor='val_accuracy', patience=0)]
nn_model.fit(X_train_seq, y_train, batch_size=128, epochs=12, validation_split=0.2, callbacks=callbacks)
```

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/429e72fa-4e57-46fc-aec4-b871d962998c/Screen_Shot_2021-02-11_at_8.28.02_PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/429e72fa-4e57-46fc-aec4-b871d962998c/Screen_Shot_2021-02-11_at_8.28.02_PM.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/dee9b7ab-3823-4d8e-8323-3e6fc9380ec6/Screen_Shot_2021-02-11_at_8.28.14_PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/dee9b7ab-3823-4d8e-8323-3e6fc9380ec6/Screen_Shot_2021-02-11_at_8.28.14_PM.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/58a36124-51fa-4563-8239-db17e560e640/Screen_Shot_2021-02-11_at_8.32.28_PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/58a36124-51fa-4563-8239-db17e560e640/Screen_Shot_2021-02-11_at_8.32.28_PM.png)

Although we had reached similar accuracies with the logistic regression model, the NN model was way more consistent. With the logistic regression, every run through would give very different results. Also, the CountVectorizer for the logreg was problematic due to its runtime, this was improved with the NN. 

We ended up with a model accuracy of 76%.

To test the model we are evaluated the sentiment as a continuous variable (how positive/negative). We could set a threshold at for example 0.5 to determine the sentiment.

A value closer to 0 means that the tweet is negative and closer to 1 means positive. 

- Example of a tweet with positive and negative:

```
input: I love you so much but I dont like this
output: 0.5572156
```

- Example of a tweet that would be neutral

```
input: I think that I will go to California next week
output: 0.5061608
```

- Example of a very negative tweet:

```
input: His speech was disgusting. I really don't agree with this horrible behaviour
output: 0.30823016
```

- Example of a tweet we expect to be very positive

```
input: The president in Colombia is the best, I would vote for him again
output: 0.81856334
```

- Example of a tweet we would expect to be neutral

```
input: I read an article today
output: 0.6176232
```

## Application & Twitter Integration

To make our model useful, we developed a Flask App to be run locally, where a user can input a topic of interest and receive statistics about sentiment surrounding the topic as well as the most extreme positive/negative tweets in that topic. 

We developed a main page using HTML with W3-css properties, with a form where a user can input their topic of interest. On submission, the topic is registered 

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fe981d21-211b-4ba5-ac2e-fd246948b54b/Screen_Shot_2021-02-25_at_12.31.33_PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fe981d21-211b-4ba5-ac2e-fd246948b54b/Screen_Shot_2021-02-25_at_12.31.33_PM.png)

Once we have a topic, we use it as a query parameter in the Twitter Developer API (which we created an account for) 

Code to get tweets (after authentication has been done)

```python
def get_tweets(access_token, topic):
    search_headers = {
        'Authorization': 'Bearer {}'.format(access_token)
    }

    search_params = {
        'q': topic,
        'result_type': 'recent',
        'count': NUM_TWEETS #variable used to ensure we don't exceed daily limits in Twitter API 
    }

    search_url = '{}1.1/search/tweets.json'.format(BASE_URL)

    search_resp = requests.get(search_url, headers=search_headers, params=search_params)
    tweet_data = search_resp.json()

    return [t['text'] for t in tweet_data['statuses']]
```

Once the NUM_TWEETS tweets are obtained, the flask app uses the tokenizer and neural network model on sentiment analysis to get the sentiment for each tweet, the average sentiment, and the most positive/negative ones

In the future, we would reformat the code to use an OOP approach, but we wrote simple functions as the API was quite simple

```python
def prepare_tweet(tweet):
    tokenizer = pickle.load(open("tokenizer.pkl", 'rb'))
    return sequence.pad_sequences(tokenizer.texts_to_sequences(tweet), maxlen=100, value=0)

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
```

The resulting page: 

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f3308142-1fa2-4206-83d0-83b65e4667bc/Screen_Shot_2021-02-24_at_5.51.57_PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f3308142-1fa2-4206-83d0-83b65e4667bc/Screen_Shot_2021-02-24_at_5.51.57_PM.png)

## Challenges and future works

- This model will still be improved, we could into using other layers and methods of preprocessing for improval, but for now, it is providing satisfactory results. On kaggle, the highest accuracy we found for this specific dataset was of 79%, which is just 2% higher than our results.
- Runtime: training the model on all the dataset was taking way too long. Our solution was to reduce the size of the training set.
- Analyzing hashtags and LDA topic modelling did not deliver good results. More investigation could be invested in those areas to  try to improve the model and add more interesting functionalities.
