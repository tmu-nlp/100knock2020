from collections import defaultdict

class mecab():
    def __init__(self, line):
        line = line.split("\t")
        if len(line) > 1:
            attr = line[1].split(",")
            self.surface = line[0]
            self.base = attr[-3]
            self.pos = attr[0]
            self.pos1 = attr[1]
            self.EOS = False
        else:
            self.EOS = True

    def extractpos_surface(self, param):
        if not self.EOS:
            if self.pos == param:
                return self.surface

    def extractpos_base(self, param):
        if not self.EOS:
            if self.pos == param:
                return self.base

def morph2sent(listmecab):
    sentdict = {}
    sent = []
    BOS = True
    for elem in listmecab:
        if not elem.EOS:
            sent.append((elem.surface, elem))
        elif BOS and elem.EOS:
            sentdict["".join([elem[0] for elem in sent])] = sent
            BOS = False
        else:
            sent = []
            BOS = True
    return sentdict
        

if __name__ == "__main__":
    with open("neko.txt.mecab", "r") as neko:
        neko = neko.readlines()
    
    """
    for i in range(1, 10):
        nekko = mecab(neko[i])
        if not nekko.EOS:
            print(nekko.surface)
    
    nekos = []
    for i in range(len(neko)):
        nekos.append(mecab(neko[i]))
    
    print(morph2sent(nekos))
    """