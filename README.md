## Tweet generator made with a LSTM neural network trained with scraped tweets 

- [tweet_scraper.py](./tweet_scraper.py) Scrapes tweets from a given account and stores them in a csv file. 
- [train.py](./train.py) Cleans and Tokenizes the tweets and trains the LSTM.
- [generate_tweets.py](./generate_tweets.py) Genereates new tweets using a random seed text and the already trained model.

#### Example:
    Seed Text:  mesmo assim movimento que não contava com minha concordancia ja que não queria voltar para trabalhar em governo\n6\n10\n17 plenário da \n@camaradeputados\n aprova
    Generated Text:  quarta brasil vem hoje mais o ele atribuída que ele para desmentir hoje\n32\n371\n296 às utilização\n4\n92\n62 já estadao que todos so uma legislativo dia na estudo

