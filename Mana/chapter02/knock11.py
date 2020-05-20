filename = 'popular-names.txt'

file = open(filename, 'r')
fileLines = file.read().replace('\t', ' ')

file = open(filename, 'w')
file.write(fileLines)

# Is this work?
# sed -e 's/" "/" "/g' popular-names.txt > popular-names.txt
# expand popular-names.txt