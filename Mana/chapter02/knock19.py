file = open('popular-names.txt', 'r')
lines = [line.strip() for line in file.readlines()]

dict_line = {}

for line in lines:
    if line[0] not in dict_line:
        dict_line[line[0]] = 1
    else:
        dict_line[line[0]] = dict_line[line[0]] + 1

print(sorted(dict_line.items(), key=lambda x: x[1], reverse=True))

#uniq -c : add count at the beginning of line
#cut -f 1 -d " " popular-names.txt | sort | uniq -c | sort -k 1r (-o [output file name])