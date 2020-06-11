from knock30 import getdata
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

if __name__ == "__main__":
    mecab_file_path = "neko.txt.mecab"
    document = getdata(mecab_file_path)
    co_occurrence = defaultdict(lambda: 0)
    positions = ["名詞", "動詞", "形容詞", "副詞"]
    for sentence in document:
        flag = False
        for word in sentence:
            if word["surface"] == "猫":
                flag = True
                break
        if not flag:
            continue
        for word in sentence:
            if (word["pos"] in positions) and (word["surface"] != "猫"):
                co_occurrence[word["base"]] += 1
    x = []
    y = []
    for key, value in sorted(co_occurrence.items(), key=lambda x:x[1], reverse=True):
        if key == "*":
            continue
        x.append(key)
        y.append(value)
        if len(x) >= 10:
            break
    fp = FontProperties(fname=r"/Users/aomi/Library/Fonts/ipaexg.ttf")
    plt.bar(x, y)
    plt.xticks(fontproperties=fp)
    plt.xlabel('猫と共起頻度が高い10語', fontproperties=fp)
    plt.show()
