# 14. 先頭からN行を出力
# 自然数Nをコマンドライン引数などの手段で受け取り，入力のうち先頭のN行だけを表示せよ．
# 確認にはheadコマンドを用いよ．

import sys

def my_head(input_file_path, n):
    count = 0
    with open(input_file_path, "r") as input_file:
        for line in input_file:
            line = line.strip()
            print(f"{line}")
            count += 1
            if count >= n:
                return

if __name__ == "__main__":
    N = int(sys.argv[1])
    input_file_path = "popular-names.txt"
    my_head(input_file_path, N)

# コマンドで実行
# head popular-names.txt