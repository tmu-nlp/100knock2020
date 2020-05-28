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

dic_list = sorted(dic.items(), key = lambda x:(x[1], x[0]), reverse = True)

for name, freq in dic_list:
    freq = str(freq)
    print("{} {}".format(freq.rjust(7), name))

#diff -s <(python3 knock19.py) <(cut --fields 1 popular-names.txt | sort | uniq --count | sort - k 1, 2 - -reverse)
