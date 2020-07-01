'''
61. 単語の類似度Permalink
“United States”と”U.S.”のコサイン類似度を計算せよ
'''
import pickle

with open("./models/google_model.pickle", mode="rb") as f:
    model = pickle.load(f)

print(model.similarity("United_States", "U.S."))

'''
0.73107743
'''