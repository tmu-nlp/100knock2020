"""
64. アナロジーデータでの実験Permalink
単語アナロジーの評価データをダウンロードし，
vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)を計算し，
そのベクトルと類似度が最も高い単語と，その類似度を求めよ．
求めた単語と類似度は，各事例の末尾に追記せよ．
"""

import pickle
from gensim.models import word2vec

analogy_file = "analogy_data"
ans_file = "64_ans_file"
my_ans_file = "64_my_ans_file"

model_file = "model.sav"
with open(model_file, "rb") as file_model:
    model = pickle.load(file_model)

if __name__ == "__main__":
    with open(analogy_file, encoding="utf-8") as file_analogy, \
            open(ans_file, "w", encoding="utf-8") as file_ans, \
            open(my_ans_file, "w", encoding="utf-8") as file_my_ans:
        for line in file_analogy:
            words = line.split()
            if words[0] == ":":
                category = words[1]
                continue
            vec1, vec2, vec3, vec4 = words
            results = model.wv.most_similar(positive=[vec2, vec3], negative=[vec1])
            file_ans.write(f"{vec4}\n")
            file_my_ans.write(f"{category}\t{results[0]}\n")
