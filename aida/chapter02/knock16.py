# Code
from knock10 import read_file

lines = read_file()

print('please type number of split files')
print('n = ', end='')
n = int(input())
each_lines = len(lines) // n

now_id = 0
for i in range(n):
    #print(now_id, now_id+each_lines)
    file_name = f'./{i}th_bin.txt'
    with open(file_name, 'w') as fp:
        for line in lines[now_id:now_id+each_lines]:
            fp.write(f'{line}\n')
    now_id += each_lines


# Unix command
# $split -l {number} popular-names.txt
