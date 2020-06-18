import json

class Word():
    def __init__(self, surface, lemma, pos, governor = '', dep = ''):
        self.surface = surface
        self.lemma = lemma
        self.pos = pos
        self.head = governor
        self.dep = dep
        self.child = []

    def update_word(self, gov, dep, child_list):
        self.head = gov
        self.dep = dep
        self.child = child_list

    def print_words(self):
        return self.surface + "\t" + self.lemma + "\t" + self.pos

    def print_dependency(self):
        return self.surface + "\t" + self.head + "\t" + self.dep + "\t[" + ' '.join(self.child) + "]"

    def print_all(self):
        return self.surface + "\t" + self.lemma + "\t" + self.pos + "\t" + self.head + "\t" + self.dep + "\t[" + ' '.join(self.child) + "]"
    
def read_words(json_file):
    data = json.load(json_file)
    text = data['sentences']
    sentences = []
    for sent in text:
        tokens = sent['tokens']
        sent_parse = []
        for t in tokens:
            surface = t['originalText']
            lemma = t['lemma']
            pos = t['pos']
            words = Word(surface, lemma, pos)
            sent_parse.append(words)
        sentences.append(sent_parse)
    return sentences

if __name__ == "__main__":
    jsonf = open('ai.en.txt.json','r') 
    sent = read_words(jsonf)
    for s in sent[0]:
        print(s.print_words())




'''
In      in      IN
computer        computer        NN
science science NN
,       ,       ,
artificial      artificial      JJ
intelligence    intelligence    NN
(       -lrb-   -LRB-
AI      ai      NN
)       -rrb-   -RRB-
,       ,       ,
sometimes       sometimes       RB
called  call    VBN
machine machine NN
intelligence    intelligence    NN
,       ,       ,
is      be      VBZ
intelligence    intelligence    NN
demonstrated    demonstrate     VBN
by      by      IN
machines        machine NNS
,       ,       ,
in      in      IN
contrast        contrast        NN
to      to      TO
the     the     DT
natural natural JJ
intelligence    intelligence    NN
displayed       display VBN
by      by      IN
humans  human   NNS
and     and     CC
animals animal  NNS
.       .       .
'''