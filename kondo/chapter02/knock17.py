file = "popular-names.txt"

with open(file) as open_file:
    file_data = open_file.read()

file_data = file_data.splitlines()

dic = {}
for line in file_data:
    words = line.split()
    if words[0] in dic:
        dic[words[0]] += 1
    else:
        dic[words[0]] = 1

dic_list = sorted(dic.items(), key=lambda x: x[0])

for name, freq in dic_list:
    print(name)

#diff -s <(python3 knock17.py) <(cut - -fields 1 popular-names.txt | sort | uniq)
