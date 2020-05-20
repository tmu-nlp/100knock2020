file_col1 = open('col1.txt', 'r')
col1_lines = [line for line in file_col1.readlines()]
file_col2 = open('col2.txt', 'r')
col2_lines = [line for line in file_col2.readlines()]

file_merge = open('merge.txt', 'w')

for i in range(len(col1_lines)):
    file_merge.write(col1_lines[i].strip() + '\t' + col2_lines[i])

#paste col[1-2].txt > merge.txt