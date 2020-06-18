from knock40 import Word
from knock41 import read_dependency

def extract_SPO_ext(data):
    sent_spo = []    
    for sent in data:
        for s in sent:
            labels = s.print_all()
            labels = labels.split('\t')
            if labels[2] == 'VBD' or labels[2][:2] == 'NN':
                if labels[4] == 'nsubj' or labels[4] == 'dobj' or labels[4] == 'ROOT' or labels[4] == 'compound':
                    sent_spo.append(labels)
    
    res = []
    for w in range(len(sent_spo)-4):
        spoPtn = sent_spo[w:w+4]
        w1 = spoPtn[0][4] == 'compound'
        w2 = spoPtn[1][4] == 'nsubj'
        w3 = spoPtn[2][4] == 'ROOT'
        w4 = spoPtn[3][4] == 'dobj'
        if w1 and w2 and w3 and w4:
            temp = spoPtn[0][0] + ' ' + spoPtn[1][0] + ' ' + spoPtn[2][0] + ' ' + spoPtn[3][0]
            res.append(temp)
    return res

if __name__ == "__main__":
    jsonf = open('ai.en.txt.json','r') 
    text = read_dependency(jsonf)
    sent_SPO = extract_SPO_ext(text)
    for s in sent_SPO:
        print(s)

'''
Ken computers enabled advances machine
Go AlphaGo won match Ke
AI research began possibility symbol
Roger Schank described approaches knowledge
AI researchers adopted tools Markov
smartphone Google used LSTM machine
Applied Vienna opened exhibitions Ars
Wendell Wallach introduced concept Moral
MacBook-1719:chapter05 khairunnisaoryza$ python3 knock46.py
Marvin Minsky agreed intelligence
computer project inspired U.S
Ken computers enabled advances
Go AlphaGo won match
computer learning ability patterns
Machine perception ability input
Computer vision ability input
AI research began possibility
research team used results
Roger Schank described approaches
AI researchers adopted tools
sample kind came search
AI Classifiers functions pattern
decision tree machine algorithm
Frank Rosenblatt invented perceptron
Yann LeCun applied backpropagation
smartphone Google used LSTM
Applied Vienna opened exhibitions
Wendell Wallach introduced concept
David Chalmers identified problems
Commander superintelligence agent intelligence
Isaac Asimov introduced Laws
George Sorayama considered robots
'''