import MeCab

neko = "neko.txt"
neko_mecab = "neko.txt.mecab"

def morphological_analysis(data_file, analysis_file):
    with open(data_file, encoding="utf-8") as data, \
            open(analysis_file, encoding="utf-8", mode="w") as analysis:

        mecab = MeCab.Tagger()
        analysis.write(mecab.parse(data.read()))

if __name__ == "__main__":
    morphological_analysis(neko, neko_mecab)