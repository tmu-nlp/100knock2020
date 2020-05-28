with open('popular-names.txt') as f \
    , open('col1.txt', 'w') as f_1 \
    , open('col2.txt', 'w') as f_2:
    for line in f:
        line = line.replace('\t', ' ')
        list = line.split()
        f_1.write(list[0] + '\n')
        f_2.write(list[1] + '\n')

#cut -f 1 popular-names.txt #デフォルトで区切り文字はタブ文字
#cut -f 2 popular-names.txt
