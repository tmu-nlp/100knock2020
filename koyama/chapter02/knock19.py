# 19. 各行の1コラム目の文字列の出現頻度を求め，出現頻度の高い順に並べる
# 各行の1列目の文字列の出現頻度を求め，その高い順に並べて表示せよ．
# 確認にはcut, uniq, sortコマンドを用いよ．

from collections import defaultdict

def get_name_freq(input_file_path):
    name_dict = defaultdict(lambda: 0)
    with open(input_file_path, "r") as input_file:
        for line in input_file:
            line = line.strip().split()
            name_dict[line[0]] += 1
    return name_dict


if __name__ == "__main__":
    input_file_path = "output.txt"
    name_dict = get_name_freq(input_file_path)
    for key, value in sorted(name_dict.items(), key=lambda x:x[1], reverse=True):
        print(f"{key}\t{value}")

# コマンドで実行
# cut -f 1 popular-names.txt | sort | uniq -c | sort -n -r