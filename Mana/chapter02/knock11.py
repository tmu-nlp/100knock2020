filename = 'popular-names.txt'

file = open(filename, 'r')
fileLines = file.read().replace('\t', ' ')

file = open(filename, 'w')
file.write(fileLines)

# expand -t 1 popular-names.txt