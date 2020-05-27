N = int(input('natural number : '))
with open('popular-names.txt') as file:
    cnt = 0
    for line in file:
        if cnt < N:
            print(line, end = '')
            cnt += 1
        else: break

#head -n 4 popular-names.txt
