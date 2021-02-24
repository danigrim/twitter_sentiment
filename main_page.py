page_html = '''
        <!DOCTYPE html>
        <html>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
        <body style="background: Khaki">
        <div class="w3-container">
	    <header class="w3-container w3-padding-32 w3-center w3-#c1c1f0" id="home">
            <h1 class="w3-jlarge"><span class="w3-hide-small" style="font-family: Monospace">Twitter Polarization Project </span></h1>
            <p style="font-family: Monospace"> Find out the latest opinions on your topics of interest </p>
            </head>
                <div class="search-container">
                    <form action = "http://127.0.0.1:5000/get_pred" method = "post">
                        <label for="topic">Topic of Interest:</label><br>
                        <p><input class="w3-input w3-padding-16" type="text" placeholder="Example: Israel" name="topic"></p>
                        <button type="submit"> Get Current Sentiment on Topic </button>
                    </form>
                </div>
                </div>
                </body>
                </html>
                '''


def make_result_page(topic, average_sentiment, pos_tweet, neg_tweet):
    pg = f"""
 <!DOCTYPE html>
    <html>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
        <body style="background: Khaki">
        <div class="w3-container">
	    <header class="w3-container w3-padding-32 w3-center w3-#c1c1f0" id="home">
            <h1 class="w3-jlarge"><span class="w3-hide-small" style="font-family: Monospace">Twitter Polarization Project </span></h1>
            <p style="font-family: Monospace"> Find out the latest opinions on your topics of interest </p>
            </head>
        <title>Results: </title>
            <div class="w3-container">

            <h2>Sentiment Scale for {topic}: </h2>
             <p style="font-family: Monospace">The scale shows the average % of positivity in the tweets </p>
                <p style="font-family: Monospace">0% is the most negative, 100% the most positive</p>
    <div class="w3-border">
    <div class="w3-green" style="height:24px;width:{int(average_sentiment)}%">{int(average_sentiment)}%</div>
    </div><br>
    <p> The most positive tweet was: {pos_tweet} </p>
    <p> The most negative tweet was: {neg_tweet} </p>
    </div>
    </body>
    </html>
    """
    return pg