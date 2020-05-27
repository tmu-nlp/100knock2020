with open('col1.txt') as f_1 \
    , open('col2.txt') as f_2 \
    , open('column.txt', 'w') as col_m:
    for line_1, line_2 in zip(f_1, f_2):
        col1 = line_1.replace('\n', '')
        col2 = line_2.replace('\n', '')
        line = col1 + '\t' + col2
        col_m.write(line + '\n')

#paste col1.txt col2.txt #デフォルトで区切り文字はタブ文字, 列方向に連結
