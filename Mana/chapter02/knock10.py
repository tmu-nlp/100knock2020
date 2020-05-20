file = open('popular-names.txt', 'r')
lines = [line for line in file.readlines()]

print(len(lines))

#wc = word count, option: -l = line length
#wc -l popular-names.txt