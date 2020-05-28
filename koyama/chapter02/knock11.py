# 11. タブをスペースに置換
# タブ1文字につきスペース1文字に置換せよ．
# 確認にはsedコマンド，trコマンド，もしくはexpandコマンドを用いよ．

def my_tr(input_file_path):
    with open(input_file_path, 'r') as input_file:
        for line in input_file:
            line = line.replace('\t', ' ').strip()
            print(line)

if __name__ == "__main__":
    input_file_path = "./popular-names.txt"
    my_tr(input_file_path)

# コマンドで実行
# cat popular-names.txt | tr '\t' ' '