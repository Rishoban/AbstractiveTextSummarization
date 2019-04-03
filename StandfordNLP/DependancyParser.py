from nltk.parse import CoreNLPParser
from nltk.parse.corenlp import CoreNLPDependencyParser
dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')

parse, = dep_parser.raw_parse('The quick brown fox jumps over the lazy dog.')

(tom_sen1, ), (tom_sen2, ), (tom_sen3, ) = dep_parser.raw_parse_sents(
        ['Tom Sawyer went to town.',
         'He met a friend.',
         'Tom was happy.'])
