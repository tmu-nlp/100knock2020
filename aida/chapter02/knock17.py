# Code
from knock10 import read_file

lines = read_file()

word_list = []
for line in lines:
    words = line.split('\t')
    word_list.append(words[0])
#word_list = [line.split('\t')[0] for line in lines]
word_list = set(word_list)

for uniq_word in word_list:
    print(uniq_word)

# Unix command
# $cut -f 1 popular-names.txt | sort | uniq
