import MeCab
import re

analysis_file = "neko.txt.mecab"

def data_mapping(file):
    with open(file, encoding="utf-8") as data:
        morphemes = []
        line = data.readline()
        while(line):
            result = re.split('[,\t\n]', line)
            result = result[:-1] #\n消すとリストの最後にから文字が入って気持ち悪い
            line = data.readline()
            if len(result) < 2: #最後の EOS\n の処理
                continue
            morpheme = {
                'surface': result[0],
                'base': result[7],
                'pos': result[1],
                'pos1': result[2],
            }
            morphemes.append(morpheme)
            if result[0] == "。":
                #ジェネレーターを利用。句点までのデータを生成する。
                yield morphemes
                morphemes = []

if __name__ == "__main__":
    lines = data_mapping(analysis_file)
    t = 0
    for line in lines: #yeildの数だけ実行。生成されたものを返す。
        print(line)
        t+=1
        if t == 10:
            break