sum = 0
with open('popular-names.txt') as f:
    for line in f:
        sum += 1

print('行数 : ', sum)

#wc -l popular-names.txt
