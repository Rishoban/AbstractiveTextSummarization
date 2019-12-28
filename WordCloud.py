from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
import nltk
from nltk.stem import PorterStemmer 

file = open("Sources.txt", "r", encoding = "utf-8")
filedata = file.readlines()

ps = PorterStemmer() 
stock_keywords = ['annual', 'report', 'arbitrag', 'averag', 'down', 'bear', 'market', 'beta', 'sharehold', 'strategi','sale', 'financi', 'solvenc', 'buy', 'sell', 'stock', 'invest', 'share', 'trade', 'price', 'stabl', 'dividend', 'fiscal', 'exchang', 'bours', 'bull', 'broker', 'bid', 'close', 'execut', 'high', 'index', 'ipo', 'public', 'offer', 'leverag', 'low', 'margin', 'purchas', 'minimum', 'balanc', 'margin', 'account', 'open', 'order', 'portfolio', 'ralli', 'quot', 'sector', 'spread', 'volatil', 'volum', 'yield', 'bottom', 'line','perform','revenu','loss', 'profit', 'grow', 'increas', 'decreas']

from nltk.tokenize import sent_tokenize
sentences = sent_tokenize(filedata[0])

comment_words = ' '
stopwords = set(STOPWORDS) 

for val in sentences:
    tokens = val.split()
    filter_keywords = []
    for i in range(len(tokens)): 
        f = ps.stem(tokens[i])
        if f in stock_keywords:
            filter_keywords.append(tokens[i])
        #tokens[i] = tokens[i].lower() 
          
    for words in filter_keywords: 
        comment_words = comment_words + words + ' '
        
wordcloud = WordCloud(width = 1000, height = 1000, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
    