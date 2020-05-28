# 18. 各行を3コラム目の数値の降順にソート
# 各行を3コラム目の数値の逆順で整列せよ（注意: 各行の内容は変更せずに並び替えよ）．
# 確認にはsortコマンドを用いよ（この問題はコマンドで実行した時の結果と合わなくてもよい）．

def sort_col3(input_file_path):
    lines = []
    with open(input_file_path, "r") as input_file:
        for line in input_file:
            line = line.strip().split()
            line[2] = int(line[2])
            lines.append(line)
    lines.sort(key=lambda x:x[2], reverse=True)
    for line in lines:
        line[2] = str(line[2])
    return lines

if __name__ == "__main__":
    input_file_path = "popular-names.txt"
    lines = sort_col3(input_file_path)
    for line in lines:
        print("\t".join(line))

# コマンドで実行
# sort -n -r -k 3 popular-names.txt
