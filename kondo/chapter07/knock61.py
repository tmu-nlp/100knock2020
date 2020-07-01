"""
61. 単語の類似度Permalink
“United States”と”U.S.”のコサイン類似度を計算せよ．
"""

import pickle
import numpy as np

model_file = "model.sav"
with open(model_file, "rb") as file_model:
    model = pickle.load(file_model)

def cos_similarity(x, y):
    return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))

if __name__ == "__main__":
    print(cos_similarity(model["United_States"], model["U.S."]))

#0.7310775