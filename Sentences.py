from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
from nltk.tokenize import sent_tokenize

def read_article(file_name):
    file = open(file_name, "r")
    filedata = file.readlines()
    #Convert the document as String Object
    sentence_string = make_String(filedata)
    #Split the sentences
    sentences = sent_tokenize(sentence_string)

    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:  # ignore if both are same sentences
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

def make_String(filedata):
    sen = ""
    for x in filedata:
        sen += x
    return sen

if __name__ == "__main__":
    stop_words = stopwords.words('english')
    snetense = read_article("Article2.txt")
    similarity_matris = build_similarity_matrix(snetense, stop_words)
    print(similarity_matris)

