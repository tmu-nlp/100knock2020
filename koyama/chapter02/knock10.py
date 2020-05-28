# 10. 行数のカウント
# 行数をカウントせよ．確認にはwcコマンドを用いよ．

def my_wc(input_file_path):
    cnt = 0
    with open(input_file_path, 'r') as input_file:
        for _ in input_file:
            cnt += 1
    return cnt


if __name__ == "__main__":
    input_file_path = "./popular-names.txt"
    ans = my_wc(input_file_path)
    print(f"{ans=}")

# 実行結果
# ans=2780

# コマンドで実行
# wc -l popular-names.txt