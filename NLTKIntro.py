import nltk

from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk import ne_chunk


senten = """The main banquet hall at Shangri-La, was jam-packed with over a thousand
     invitee guests- ranging from the first citizen of the country himself, the Speaker,
     Opposition leader, politicians, clergy, representatives of the legal sector,
     law enforcement authorities, representatives of diplomatic missions, school
     children and various others. CIABOC, prior to the occasion, told media that they
     expect a crowd of ‘about 1,200’, however, the turnout seemed to exceed that
     modest number. The organizers had prepared a separate
     area for the VIPs to have lunch as it was impractical for the main ballroom to handle such a vast gathering."""

tokens = word_tokenize(senten)

frequeny_word = FreqDist()

for word in tokens:
    frequeny_word[word.lower()] += 1

Chunk_word = "The US president borrow $200"

make_token = word_tokenize(Chunk_word)
make_posTag = nltk.pos_tag(make_token)

final_tags = ne_chunk(make_posTag)