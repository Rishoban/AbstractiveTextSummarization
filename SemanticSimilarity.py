import nltk
from nltk.corpus import wordnet as wn
from nltk.collocations import * 

dog = wn.synset('dog.n.01')
cat = wn.synset('cat.n.01')

print(dog.path_similarity(cat))

text = """The main banquet hall at Shangri-La, was jam-packed with over a thousand
     invitee guests- ranging from the first citizen of the country himself, the Speaker,
     Opposition leader, politicians, clergy, representatives of the legal sector,
     law enforcement authorities, representatives of diplomatic missions, school
     children and various others. CIABOC, prior to the occasion, told media that they
     expect a crowd of ‘about 1,200’, however, the turnout seemed to exceed that
     modest number. The organizers had prepared a separate
     area for the VIPs to have lunch as it was impractical for the main ballroom to handle such a vast gathering."""
     
bigram_measure = nltk.collocations.BigramAssocMeasures()

finder = BigramCollocationFinder.from_words(text)
finder.nbest(bigram_measure.pmi,10)
finder.apply_freq_filter(10)


