#https://qiita.com/fantm21/items/6df776d99356ef6d14d4
#lambda 引数1,引数2,...,引数N : 返したい答えを求める計算式
with open('popular-names.txt') as file:
    num_of_ppls = []
    dic = {}
    lines = file.readlines()
    for line in lines:
        words = line.split('\t')
        num_of_ppls.append(words[2].strip('\n')) #人数のリスト

    for key, value in enumerate(num_of_ppls): #人数とインデックスの辞書
        dic[key] = int(value)

    items_sorted = sorted(dic.items(), reverse = True, key = lambda x : x[1]) #人数でソート
    for item in items_sorted:
        #item[0]はインデックス
        print(lines[item[0]], end = '')

#sort -rnk3 -t $'\t' popular-names.txt #-r:逆順, -n:数値順, -k:フィールド指定, -t:区切り文字