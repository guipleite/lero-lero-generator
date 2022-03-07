## Tweet generator made with a LSTM neural network trained with scraped tweets 

- [tweet_scraper.py](./tweet_scraper.py) Scrapes tweets from a given account and stores them in a csv file. 
- [train.py](./train.py) Cleans and Tokenizes the tweets and trains the LSTM.
- [generate_tweets.py](./generate_tweets.py) Genereates new tweets using a random seed text and the already trained model.

#### Examples:
    Seed Text:  mesmo assim movimento que não contava com minha concordancia ja que não queria voltar para trabalhar em governo\n6\n10\n17 plenário da \n@camaradeputados\n aprova
    Generated Text:  quarta brasil vem hoje mais o ele atribuída que ele para desmentir hoje\n32\n371\n296 às utilização\n4\n92\n62 já estadao que todos so uma legislativo dia na estudo

    Seed Text:  bispo manoel ferreira \ndeus me colocou aqui para continuar luta em defesa princípios evangélicos vou honrar ensinamentos cristãos compromisso que tenho com bispo quem
    Generated Text:  tenho como segundo pai que tanto me apoia\n9\n3\n13 da seja divulgaram andrade.\n1\n6\n19 meu fato crencas\n

    Seed Text:  suposto diálogo com líder da bancada absolutamente mentiroso ele simplesmente não existiu não interferi junto bancada\n9\n28\n31 ' ' bom dia todos leiam artigo líder
    Generated Text:  psd rogerio rosso em o globo hj é isso mais uma referente mim desafio ele acredito com din abraços.\n4\n23\n13 ' ' nunca falei que dep andré tivesse gravado sobre assunto que
