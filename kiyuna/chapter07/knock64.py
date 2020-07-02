"""
64. アナロジーデータでの実験
単語アナロジーの評価データをダウンロードし，
vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)を計算し，
そのベクトルと類似度が最も高い単語と，その類似度を求めよ．
求めた単語と類似度は，各事例の末尾に追記せよ．

[Ref]
- https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#word2vec-demo

[Command]
wget -qN http://download.tensorflow.org/data/questions-words.txt

[MEMO]
2015 年版の knock91-92 に対応
"""
import os
import sys
import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.pickle import load  # noqa: E402 isort:skip


def gen_questions_words(path="./questions-words.txt"):
    with open(path) as f:
        for line in f:
            if line.startswith(":"):
                yield line, None, None, None
            else:
                yield line.split()


if __name__ == "__main__":
    wv = load("chap07-embeddings")
    with open("out64.txt", "w") as f:  # out64 が消える事故を防ぐため名前を変えた
        res = []
        for w1, w2, w3, w4 in tqdm.tqdm(gen_questions_words()):
            if w2:
                [(w5, sim)] = wv.most_similar(positive=[w2, w3], negative=[w1], topn=1)
                res.append(f"{w1} {w2} {w3} {w4} {w5} {sim}\n")
            else:
                res.append(w1)
        f.writelines(res)


"""head out64
: capital-common-countries
Athens Greece Baghdad Iraq Iraqi 0.6351870894432068
Athens Greece Bangkok Thailand Thailand 0.7137669324874878
Athens Greece Beijing China China 0.7235777974128723
Athens Greece Berlin Germany Germany 0.6734622120857239
Athens Greece Bern Switzerland Switzerland 0.4919748306274414
Athens Greece Cairo Egypt Egypt 0.7527809739112854
Athens Greece Canberra Australia Australia 0.583732545375824
Athens Greece Hanoi Vietnam Viet_Nam 0.6276341676712036
Athens Greece Havana Cuba Cuba 0.6460992097854614
"""
