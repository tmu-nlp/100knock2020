# 16. ファイルをN分割する
# 自然数Nをコマンドライン引数などの手段で受け取り，入力のファイルを行単位でN分割せよ．
# 同様の処理をsplitコマンドで実現せよ．

import sys
from knock10 import my_wc

def make_output(output_file_path, output_list):
    with open(output_file_path, "w") as ouput_file:
        for line in output_list:
            print(f"{line}", file=ouput_file)

def my_split(input_file_path, n):
    n_lines = my_wc(input_file_path)
    q = n_lines // n
    r = n_lines % n
    output_list = []
    count = 1
    with open(input_file_path, "r") as input_file:
        for line in input_file:
            line = line.strip()
            output_list.append(line)
            if len(output_list) == q and r == 0:
                make_output(f"file{count}", output_list)
                output_list = []
                count += 1
            elif len(output_list) == q + 1:
                make_output(f"file{count}", output_list)
                output_list = []
                count += 1
                r -= 1

if __name__ == "__main__":
    N = int(sys.argv[1])
    input_file_path = "popular-names.txt"
    my_split(input_file_path, N)

# コマンドで実行
# split -n 10 popular-names.txt split_file