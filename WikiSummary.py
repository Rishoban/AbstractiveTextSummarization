import numpy as np
import nltk
nltk.download('punkt') # one time execution
import csv
import itertools
import networkx as nx
import math
from nltk.stem import PorterStemmer

#Read this documents
file = open("Evaluation/achu5.txt", "r", encoding = "utf-8")

filedata = file.readlines()

stock_keywords = ['gain','annual', 'report', 'arbitrag', 'averag', 'down', 'bear', 'market', 'beta', 'sharehold', 'manag', 'strategi','sale', 'financi', 'solvenc', 'buy', 'sell', 'stock', 'invest', 'share', 'trade', 'price', 'stabl', 'dividend', 'fiscal', 'exchang', 'bourse', 'bull', 'broker', 'bid', 'close', 'execut', 'high', 'index', 'ipo', 'public', 'offer', 'leverag', 'low', 'margin', 'purchas', 'minimum', 'balanc', 'margin', 'account', 'open', 'order', 'portfolio', 'ralli', 'quot', 'sector', 'spread', 'volatil', 'volum', 'yield', 'bottom', 'line','perform','revenu','loss', 'profit', 'grow', 'increas', 'decreas', 'multipl', 'roe', 'roa', 'p/e',  'alpha', 'rel',  'nasdaq',  'msci', 'hangseng', 'world',  'indic',  'ep',  'quarterli', 'forward', 'contract', 'profit', 'take', 'equiti', 'market']

Multiple_keys = ['nikkei 225', 'forward P/BV', 'Dividend Yield', 'Penny stocks', 'Value stocks', 'Growth stocks', 'risk adjusted return', 'mean reverting', 'S&P 500', 'FTSE 100', 'MSCI Emerging markets', 'technical charts', 'moving averages', 'book value', 'EBITDA growth', 'EBITDA margin', 'all time high', 'all time low', 'price gains', 'Earnings exceeding forecasts', 'Last Twelve Months', 'intrinsic value', 'upside potential', 'stock futures']
from nltk.tokenize import sent_tokenize, word_tokenize
sentences = []
for g in filedata:
   sentence = sent_tokenize(g) 
   sentences += sentence

#Read the glove file 
word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
nltk.download('stopwords')

def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

def stockKey_calculation(sen):
    count = 0
    words = word_tokenize(sen)
    keys = []
    ps = PorterStemmer()
    stem_words = []
    for e in words:
        g = ps.stem(e)
        stem_words.append(g)
    for word in stem_words:
        if word in stock_keywords:
            count += 1
            keys.append(word)
            
    for kes in Multiple_keys:
        if kes in sen:
            count += 1
            keys.append(kes)
    return count

def calculate_keys(list_keys):
    total_count = sum(list_keys)
    ratio_keys = []
    
    if total_count == 0:
        return ratio_keys
    
    for stock_num in list_keys:
        ratios = stock_num/total_count
        ratio_keys.append(ratios)
        
    return ratio_keys

def inverseRank_generator(g, smatrix):
    nodes_list = list(g.nodes)
    final_dict = {}
    for x in nodes_list:
        rank = 0
        for sim_value in nodes_list:
            rank += smatrix[x,sim_value]
        final_dict[x] = rank

    nx.set_node_attributes(g, final_dict, 'inverseRank')   
    return g, final_dict

def graph_reduction(g, rankDict, smatrix):
    key_max = max(rankDict.keys(), key=(lambda k: rankDict[k]))   #Find node which has maximum inverse rank
    g.remove_node(key_max)     #Remove node from the graph
    rest_rank = nx.get_node_attributes(g, 'inverseRank')  #get the inverse rank of the rest of the nodes
    key_max2 = min(rest_rank.keys(), key=(lambda k: rest_rank[k]))   #Find minimum inverse rank node
    connected_component = nx.node_connected_component(g, key_max2)    #Pick the suggraph whish has minimum inverse rank
    s = list(connected_component)   #Convert the set of nodes to list of nodes
    s_max = max(s)

    G_ex = nx.Graph()
    G_ex.add_nodes_from(s)
    G_ex.add_edges_from(itertools.combinations(s, 2)) #Generate graph from existing nodes and add weights
    new_matrix = np.zeros([s_max+1, s_max+1])
    for x in s:
        for y in s:
            new_matrix[x][y] = smatrix[x][y]
            G_ex.add_weighted_edges_from([(x, y, smatrix[x][y])])

    add_inverseRank, inverRank_dict = inverseRank_generator(G_ex, new_matrix) #find the inverse rank for the rest of the nodes
    return G_ex, inverRank_dict

clean_sentences = [remove_stopwords(r.split()) for r in sentences]


#Creating word vectors form each sentences
sentence_vectors = []
stockcounts = []
for i in clean_sentences:
  if len(i) != 0:
    v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
    stockcounts.append(stockKey_calculation(i))
  else:
    v = np.zeros((100,))
  sentence_vectors.append(v)

sim_mat = np.zeros([len(sentences), len(sentences)])

from sklearn.metrics.pairwise import cosine_similarity

#Cosine similarities
for i in range(len(sentences)):
  for j in range(len(sentences)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

ratio_keys = []
ratio_keys = calculate_keys(stockcounts)

mul_ratio_keys = [x * 10 for x in ratio_keys]

for val in range(len(clean_sentences)):
    sim_mat[val][val] = mul_ratio_keys[val]


G = nx.from_numpy_matrix(sim_mat)
#with open('Edges.csv', 'w') as csvfile:
#     fieldName = ['Source', 'Target', 'Weight', 'Type']
#     theWriter = csv.DictWriter(csvfile, fieldnames= fieldName)
#     theWriter.writeheader()
#    
#     for n1, n2, attr in G.edges(data=True):
#         lines = [n1, n2, attr.get('weight')]
#         if n1 == n2:
#            continue
#         else:
#             theWriter.writerow({fieldName[0]:n1, fieldName[1]: n2, fieldName[2]: attr.get('weight'),fieldName[3]:'Undirected'})


    # labels = dict((n, d['inverseRank']) for n, d in iter_G.nodes(data=True))
    # nx.draw(iter_G, labels=labels, node_size=1000)
    # pylab.show()
s1 = 'Sent'
#with open('Nodes.csv', 'w') as csvfile:
#    fieldName = ['Id', 'Label']
#    theWriter = csv.DictWriter(csvfile, fieldnames=fieldName)
#    theWriter.writeheader()
#    
#    f = list(G.nodes(data=True))
#    for node_label, attribute in f:
#        linew = [node_label, attribute.get('inverseRank')]
#        theWriter.writerow({fieldName[0]:linew[0], fieldName[1]:s1+str(node_label)})

#Using pageRank algorithm
#nx_graph = nx.from_numpy_array(sim_mat)
#scores = nx.pagerank(nx_graph)

#ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
#summary = []
#for i in range(10):
#  summary.append(ranked_sentences[i][1])


add_inverseRank, inverseRank_dict = inverseRank_generator(G, sim_mat)
num_nodes = len(G)
summary_content = math.ceil(num_nodes/2)
iter_con = num_nodes
iter_G = G
rank_dict = inverseRank_dict
while(iter_con > summary_content):
    iter_G, rank_dict = graph_reduction(iter_G, rank_dict, sim_mat)
    iter_con = len(iter_G)

print(list(iter_G.nodes(data=True)))
summary = list(iter_G.nodes)
for summary_sen in summary:
    print(sentences[summary_sen])