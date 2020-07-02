#単語アナロジーの評価データをダウンロードし，
# vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)を計算し，
# そのベクトルと類似度が最も高い単語と，その類似度を求めよ．
# 求めた単語と類似度は，各事例の末尾に追記せよ．

from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin", binary=True
)

with open("questions-words.txt", "r") as f:
  analogy = f.readlines()

ans_file = open("questions-words_ans.txt", "w")
for line in analogy:
  line = line.strip().split()
  if len(line) == 4:
    ans = model.most_similar(positive=[line[1], line[2]], negative=[line[0]], topn=1)
    ans_file.write(line[0] + " " + line[1] + " " + line[2] + " " + line[3] + " " + ans[0][0] + " " + str(ans[0][1]) + "\n")
  #print(ans[0][0], ans[0][1])
ans_file.close()