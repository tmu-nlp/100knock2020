name_list = []
name_set = set()
dic = {}

with open('popular-names.txt') as file:
    lines = file.readlines()

    for line in lines:
        name = line.split('\t')[0]
        name_list.append(name)
        name_set.add(name)

    for name in name_set:
        dic[name] = name_list.count(name) #count():指定した要素がリストにいくつ含まれているか

    items_sorted = sorted(dic.items(), reverse = True, key = lambda x : x[1])
    for item in items_sorted:
        print('{:4d} {}'.format(item[1], item[0]))
#cut -f1 popular-names.txt | sort | uniq -c | sort -rn
#uniq:-c:重複行を数える
