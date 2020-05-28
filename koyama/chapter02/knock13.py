# 13. col1.txtとcol2.txtをマージPermalink
# 12で作ったcol1.txtとcol2.txtを結合し，元のファイルの1列目と2列目をタブ区切りで並べたテキストファイルを作成せよ．
# 確認にはpasteコマンドを用いよ．

def my_paste(input_file_path1,input_file_path2, output_file_path):
    with open(input_file_path1, "r") as input_file1, open(input_file_path2, "r") as input_file2, open(output_file_path, "w") as ouput_file:
        for line1, line2 in zip(input_file1, input_file2):
            line1 = line1.strip()
            line2 = line2.strip()
            print(f"{line1}\t{line2}", file=ouput_file)

if __name__ == "__main__":
    input_file_path1 = "col1.txt"
    input_file_path2 = "col2.txt"
    output_file_path = "col1-2.txt"
    my_paste(input_file_path1, input_file_path2, output_file_path)

# コマンドで実行
# paste col1.txt col2.txt