import pprint
from knock40 import Morph
from knock40 import morph2sent

class Chunk():
    def __init__(self, morphlist):
        self.morphs = morphlist[1:]
        #self.morphs = [elem.show() for elem in morphlist[1:]]見やすい
        self.meta = morphlist[0].show()
        self.dst = self.meta[2]
        self.srcs = self.meta[1]

    def show_bunsetsu_tag(self):
        return "".join([elem.show() for elem in self.morphs])


def morph2chunk(morphlists):
    morphlists.append(Morph("*"))
    chunks = []
    chunk = []
    for elem in morphlists:
        #print(elem.show())
        if elem.show()[0] != "*" :
            chunk.append(elem)
        else:
            if chunk != []:
                chunks.append(Chunk(chunk))
            chunk = []
            chunk.append(elem)
    return chunks

if __name__ == "__main__":
    with open("ai.ja.txt.parsed", "r") as ai:
        ai = ai.readlines()

    ai_morphs = []
    for i in range(len(ai)):
        ai_morphs.append(Morph(ai[i]))
    
    #print(morph2sent(ai_morphs)[1][0].show())

    dep = morph2chunk(morph2sent(ai_morphs)[1])
    pprint.pprint([elem.show_bunsetsu_tag() for elem in dep])

"""
出力こんな感じになりますけど。
[('17D', '0', ['人工', '知能']),
 ('17D', '1', ['（', 'じん', 'こうち', 'のう', '、', '、']),
 ('3D', '2', ['AI']),
 ('17D', '3', ['〈', 'エーアイ', '〉', '）', 'と', 'は', '、']),
 ('5D', '4', ['「', '『', '計算']),
 ('9D', '5', ['（', '）', '』', 'という']),
 ('9D', '6', ['概念', 'と']),
 ('8D', '7', ['『', 'コンピュータ']),
 ('9D', '8', ['（', '）', '』', 'という']),
 ('10D', '9', ['道具', 'を']),
 ('12D', '10', ['用い', 'て']),
 ('12D', '11', ['『', '知能', '』', 'を']),
 ('13D', '12', ['研究', 'する']),
 ('14D', '13', ['計算', '機', '科学']),
 ('15D', '14', ['（', '）', 'の']),
 ('16D', '15', ['一', '分野', '」', 'を']),
 ('17D', '16', ['指す']),
 ('34D', '17', ['語', '。']),
 ('20D', '18', ['「', '言語', 'の']),
 ('20D', '19', ['理解', 'や']),
 ('21D', '20', ['推論', '、']),
 ('22D', '21', ['問題', '解決', 'など', 'の']),
 ('24D', '22', ['知的', '行動', 'を']),
 ('24D', '23', ['人間', 'に']),
 ('26D', '24', ['代わっ', 'て']),
 ('26D', '25', ['コンピューター', 'に']),
 ('27D', '26', ['行わ', 'せる']),
 ('34D', '27', ['技術', '」', '、', 'または', '、']),
 ('29D', '28', ['「', '計算', '機']),
 ('31D', '29', ['（', 'コンピュータ', '）', 'による']),
 ('31D', '30', ['知的', 'な']),
 ('33D', '31', ['情報処理', 'システム', 'の']),
 ('33D', '32', ['設計', 'や']),
 ('34D', '33', ['実現', 'に関する']),
 ('35D', '34', ['研究', '分野', '」', 'と', 'も']),
 ('-1D', '35', ['さ', 'れる', '。'])]
"""