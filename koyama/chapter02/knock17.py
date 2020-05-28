# 17. １列目の文字列の異なり
# 1列目の文字列の種類（異なる文字列の集合）を求めよ．
# 確認にはcut, sort, uniqコマンドを用いよ．

def get_names(input_file_path):
    names = set()
    with open(input_file_path, "r") as input_file:
        for line in input_file:
            line = line.strip().split()
            names.add(line[0])
    return names

if __name__ == "__main__":
    input_file_path = "popular-names.txt"
    names = get_names(input_file_path)
    for line in sorted(names):
        print(f"{line}")

# コマンドで実行
# cut -f 1 popular-names.txt | sort | uniq