"""
66. WordSimilarity-353での評価
The WordSimilarity-353 Test Collectionの評価データをダウンロードし，
単語ベクトルにより計算される類似度のランキングと，
人間の類似度判定のランキングの間のスピアマン相関係数を計算せよ．

[Ref]
- https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#word2vec-demo

[Command]
wget -qN http://www.gabrilovich.com/resources/data/wordsim353/wordsim353.zip

[MEMO]
2015 年版の knock94-95 に対応
"""
import os
import sys
from zipfile import ZipFile

from scipy.stats import spearmanr

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.message import message, Renderer  # noqa: E402 isort:skip
from kiyuna.utils.pickle import load  # noqa: E402 isort:skip


if __name__ == "__main__":
    wv = load("chap07-embeddings")
    preds, labels = [], []
    with ZipFile("wordsim353.zip") as myzip:
        message(myzip.infolist())
        with myzip.open("combined.csv") as myfile:
            myfile = map(lambda x: x.decode(), myfile)
            message("[header]", next(myfile))  # Word 1,Word 2,Human (mean)
            for line in myfile:
                word1, word2, human = line.split(",")
                preds.append(wv.similarity(word1, word2))
                labels.append(human)
    with Renderer("knock66") as out:
        out.result("Spearman corr", spearmanr(preds, labels)[0])


"""result
0.6849564489532376
"""
