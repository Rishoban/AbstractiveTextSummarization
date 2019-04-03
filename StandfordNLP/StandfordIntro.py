from nltk.parse import CoreNLPParser
from nltk.parse.corenlp import CoreNLPDependencyParser

parser = CoreNLPParser(url='http://localhost:9000')
list(parser.parse('What is the airspeed of an unladen swallow ?'.split()))

list(parser.raw_parse('What is the airspeed of an unladen swallow ?'))

dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
parses = dep_parser.parse('What is the airspeed of an unladen swallow ?'.split())
[[(governor, dep, dependent) for governor, dep, dependent in parse.triples()] for parse in parses]

parser = CoreNLPParser(url='http://localhost:9000')
list(parser.tokenize('What is the airspeed of an unladen swallow?'))

pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')
list(pos_tagger.tag('What is the airspeed of an unladen swallow ?'.split()))

ner_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='ner')
list(ner_tagger.tag(('Rami Eid is studying at Stony Brook University in NY'.split())))