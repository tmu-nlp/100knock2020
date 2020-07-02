'''
64. アナロジーデータでの実験Permalink
単語アナロジーの評価データをダウンロードし，vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)を
計算し，そのベクトルと類似度が最も高い単語と，その類似度を求めよ．求めた単語と類似度は，各事例の末尾に追記せよ．
'''
import pickle
from pprint import pprint
from tqdm import tqdm

with open("./models/google_model.pickle", mode="rb") as f:
    model = pickle.load(f)

with open("./data/questions-words.txt") as infile, open(
    "./data/questions-words-results.txt", mode="w"
) as outfile:
    bar = tqdm(total=19558)
    for line in infile:
        if line[0] == ":":
            """
            cols = line.split()
            analogy = cols[1]
            """
            outfile.write(line)
            continue
        cols = line.strip().split()
        result = model.most_similar(
            positive=[cols[1], cols[2]], negative=[cols[0]], topn=1
        )
        # cols.insert(0, analogy)
        cols.append(result[0][0])
        cols.append(str(result[0][1]))
        outline = " ".join(cols)
        outfile.write(outline + "\n")
        bar.update(1)
