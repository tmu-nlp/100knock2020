N = int(input('natural number : '))
with open('popular-names.txt') as file:
    n = sum(1 for line in file) #行数

with open('popular-names.txt') as file:
    cnt = 0
    for line in file:
        if cnt >= n - N :
            print(line, end = '')
        else : pass
        cnt += 1

'''
with open('popular-names.txt') as file:
    lines = file.readlines()

    for line in lines[-N::]:
        print(line, end = '')
'''


#tail -n 4 popular-names.txt
