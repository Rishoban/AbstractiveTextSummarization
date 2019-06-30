from nltk.parse import CoreNLPParser
from nltk.parse.corenlp import CoreNLPDependencyParser
import networkx as nx
import matplotlib.pyplot as plt
dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')


#parse, = dep_parser.raw_parse('The quick brown fox jumps over the lazy dog.')
#G = nx.DiGraph()
#G = parse.nx_graph()
#nx.draw(G)
#plt.show()

senten = """The main banquet hall at Shangri-La, was jam-packed with over a thousand
     invitee guests- ranging from the first citizen of the country himself, the Speaker,
     Opposition leader, politicians, clergy, representatives of the legal sector,
     law enforcement authorities, representatives of diplomatic missions, school
     children and various others. CIABOC, prior to the occasion, told media that they
     expect a crowd of ‘about 1,200’, however, the turnout seemed to exceed that
     modest number. The organizers had prepared a separate
     area for the VIPs to have lunch as it was impractical for the main ballroom to handle such a vast gathering."""
     

check_parser, = dep_parser.raw_parse("""The main banquet hall at Shangri-La, was jam-packed with over a thousand
     invitee guests- ranging from the first citizen of the country himself, the Speaker,
     Opposition leader, politicians, clergy, representatives of the legal sector,
     law enforcement authorities, representatives of diplomatic missions, school
     children and various others.""")

(tom_sen1, ), (tom_sen2, ), (tom_sen3, ) = dep_parser.raw_parse_sents(
        ['Tom Sawyer went to town.',
         'He met a friend.',
         'Tom was happy.'])

for predicate, subject, objects in tom_sen1.triples():
    print(predicate)

# =============================================================================
# import subprocess
# myinput = open('Source.txt')
# myoutput = open('outs.txt', 'w')
# p = subprocess.Popen('A:\FinalYearProject\PythonNLP\senna-v3.0\senna-v3.0\senna\senna-win32.exe', stdin=myinput, stdout=myoutput)
# p.wait()
# myoutput.flush()
# =============================================================================
