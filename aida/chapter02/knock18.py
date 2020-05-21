# Code
from knock10 import read_file

lines = read_file()

id_value_dic = {}
for index, line in enumerate(lines):
    words = line.split('\t')
    value = int(words[3])
    id_value_dic[index] = value

sorted_ids = sorted(id_value_dic.items(), key=lambda x:x[1], reverse=True)

for id_value in sorted_ids:
    index = id_value[0]
    print(lines[index])

# Unix command
# $sort popular-names.txt --key=3,3 --numeric-sort --reverse
