# 15. 末尾のN行を出力Permalink
# 自然数Nをコマンドライン引数などの手段で受け取り，入力のうち末尾のN行だけを表示せよ．
# 確認にはtailコマンドを用いよ．

import sys
import queue

def my_tail(input_file_path, n):
    q = queue.Queue()
    with open(input_file_path, "r") as input_file:
        for line in input_file:
            line = line.strip()
            q.put(line)
            if q.qsize() > n:
                q.get()
    while not q.empty():
        print(f"{q.get()}")

if __name__ == "__main__":
    N = int(sys.argv[1])
    input_file_path = "popular-names.txt"
    my_tail(input_file_path, N)

# コマンドで実行
# tail popular-names.txt