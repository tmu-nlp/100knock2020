from knock40 import Word
from knock41 import read_dependency

def extract_SPO(data):
    sent_spo = []    
    for sent in data:
        for s in sent:
            labels = s.print_all()
            labels = labels.split('\t')
            if labels[2] == 'VBD' or labels[2][:2] == 'NN':
                if labels[4] == 'nsubj' or labels[4] == 'dobj' or labels[4] == 'ROOT':
                    sent_spo.append(labels)
    
    res = []
    for w in range(len(sent_spo)-3):
        spoPtn = sent_spo[w:w+3]
        w1 = spoPtn[0][4] == 'nsubj'
        w2 = spoPtn[1][4] == 'ROOT'
        w3 = spoPtn[2][4] == 'dobj'
        if w1 and w2 and w3:
            temp = spoPtn[0][0] + ' ' + spoPtn[1][0] + ' ' + spoPtn[2][0]
            res.append(temp)
    return res

if __name__ == "__main__":
    jsonf = open('ai.en.txt.json','r') 
    text = read_dependency(jsonf)
    sent_SPO = extract_SPO(text)
    for s in sent_SPO:
        print(s)

'''
Minsky agreed intelligence
Progress slowed projects
project inspired U.S
development enabled development
match defeated champions
computers enabled advances
AlphaGo won games
AlphaGo won match
Go year Google
China accelerated funding
approach analogizers records
researchers developed algorithms
algorithms proved problems
representation systems knowledge
learning ability patterns
Regression attempt function
perception ability input
vision ability input
planning process task
computing umbrella systems
projects failed limitations
number explored connection
research began possibility
Simon studied skills
team used results
laboratory focused logic
Researchers found problems
Schank described approaches
researchers began knowledge
revolution led form
component base AI
Researchers rejected AI
researchers adopted tools
language permitted level
successes led emphasis
step search path
kind came search
logics forms domains
networks tool algorithm
Classifiers functions pattern
tree machine algorithm
successes included market
Rosenblatt invented perceptron
learning network chain
publication introduced way
LeCun applied backpropagation
recognition experienced jump
Google used LSTM
LSTM improved captioning
AlphaGo brought era
derivative test Computers
type test typing
study demonstrated surgery
AICPA introduced course
benefit risk AI
exhibition Machines overview
Association dedicated issue
Vienna opened exhibitions
Scientists described goals
Musk donated research
Wallach introduced concept
Chalmers identified problems
superintelligence agent intelligence
survey showed disagreement
Asimov introduced Laws
Sorayama considered robots
'''