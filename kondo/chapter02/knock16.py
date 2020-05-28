from sys import argv, stdout

#N = int(input())

file = "popular-names.txt"

with open(file) as open_file:
    file_data = open_file.readlines()

argv[1] = int(argv[1])
#⌈a/b⌉ = (a + b - 1) / b
tnl = (len(file_data) + argv[1] - 1)//argv[1]

count = 0
now_file = -1
for index, line in enumerate(file_data):
    if now_file != index // tnl:
        if now_file >= 0:
            out_file.close()
        now_file = index // tnl
        out_file = open('y_{:02d}.txt'.format(now_file), mode='w')
    out_file.write(line)
out_file.close()

#split -n l/59 -d popular-names.txt
