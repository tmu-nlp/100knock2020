with open('popular-names.txt') as f:
    for line in f:
        line = line.replace('\t', ' ')
        print(line, end = '')

#sed $'s/\t/ /g' popular-names.txt
#cat popular-names.txt | tr '\t' ' '
#expand -t 1 popular-names.txt