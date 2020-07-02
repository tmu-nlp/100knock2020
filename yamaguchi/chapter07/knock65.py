def main():
    # knock64の結果を開く
    with open('knock64.txt', 'r') as f:
        analogies = f.readlines()

    count = 0
    correct = 0

    # 意味的アナロジーと文法的アナロジーそれぞれで，正解していたら「correct」を+1する．
    for analogy in analogies:
        words = analogy.split()

        if len(words) == 6:
            count += 1
            if (words[3] == words[4]):
                    correct += 1

    # 正解数/全体の数で正解率を計算
    rate = correct / count

    # 結果をファイルに保存
    with open('knock65.txt', mode='w', encoding="utf-8") as f:
        print(rate, file=f)

if __name__ == '__main__':
    main()
