names = set()
with open('popular-names.txt') as file:
    for line in file:
        lis = line.split('\t')
        names.add(lis[0])

names_sorted = sorted(names)
for name in names_sorted:
    print(name)
    
#cut -f 1 popular-names.txt | sort | uniq
