# 12. 1列目をcol1.txtに，2列目をcol2.txtに保存
# 各行の1列目だけを抜き出したものをcol1.txtに，2列目だけを抜き出したものをcol2.txtとしてファイルに保存せよ．
# 確認にはcutコマンドを用いよ．

def my_cut(input_file_path, output_file_path, n):
    n -= 1 # 0-indexed にする
    with open(input_file_path, "r") as input_file, open(output_file_path, "w") as output_file:
        for line in input_file:
            line = line.strip().split()
            print(f"{line[n]}", file=output_file)


if __name__ == "__main__":
    input_file_path = "popular-names.txt"
    output_file_path1 = "col1.txt"
    output_file_path2 = "col2.txt"
    my_cut(input_file_path, output_file_path1, 1)
    my_cut(input_file_path, output_file_path2, 2)

# コマンドで実行
# cut -f 1 popular-names.txt
# cut -f 2 popular-names.txt