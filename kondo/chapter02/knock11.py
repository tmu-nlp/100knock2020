file1 = "popular-names.txt"

#read()の後にsplitlines() → うまくいく
#read()の後にsplit('\n') → 要素の最後に無駄な空要素が出来る → split(',')考えれば当たり前か
with open(file1) as open_file:
    file_data = open_file.read()
file_data = file_data.splitlines()
#  print(file_data)
for line in file_data:
    print(line.replace('\t', ' '))


#readlines()を使うと各要素の末尾に'\n'が残るのが少し厄介 → print(, end = '')で解決
"""with open(file1) as open_file:
    file_data = open_file.readlines()
#  print(file_data)
for line in file_data:
    print(line.replace('\t', ' '), end = '')"""

#cat popular-names.txt | expand -t 1
#cat popular-names.txt | tr '\t' ' '
#cat popular-names.txt | sed -e 's/\t/ /g'
