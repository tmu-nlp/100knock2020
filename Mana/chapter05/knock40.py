#Morph class作成

class Morph():
    def __init__(self, line):
        line = line.split("\t")
        if line[0][0] == "*":
            self.meta = line[0].strip().split()
            self.EOS = False
        elif len(line) > 1:
            attr = line[1].split(",")
            self.surface = line[0]
            self.base = attr[-3]
            self.pos = attr[0]
            self.pos1 = attr[1]
            self.EOS = False
        else:
            self.EOS = True

    def show(self):
        if self.EOS:
            return "EOS"
        else:
            try:
                return self.surface
            except AttributeError:
                return self.meta

def morph2sent(listcabocha):
    sents = []
    sent = []
    BOS = True
    for elem in listcabocha:
        if not elem.EOS:
            sent.append(elem)
        elif BOS and elem.EOS:
            sents.append(sent)
            BOS = False
        else:
            sent = []
            BOS = True
    return sents

if __name__ == "__main__":
    with open("ai.ja.txt.parsed", "r") as ai:
        ai = ai.readlines()

    ai_morphs = []
    for i in range(len(ai)):
        ai_morphs.append(Morph(ai[i]))
    
    print(morph2sent(ai_morphs)[0])


    sents = morph2sent(ai_morphs)[2]
    for sent in sents:
        print(sent.show())
    #print(sents[1].show()[0])
