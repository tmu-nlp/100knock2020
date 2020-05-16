def n_gram(n, str):
    lens = len(str)
    list = []
    for i in range(lens - n + 1):
        list.append(str[i:i + n])

    return list

str1 = "paraparaparadise"
str2 = "paragraph"

X = n_gram(2, str1)
Y = n_gram(2, str2)

print("X: {}\nY: {}\n".format(X, Y))

set_X = set(X)
set_Y = set(Y)

union = set_X | set_Y
intersec = set_X & set_Y
diff = set_X - set_Y

print("和集合: {}\n積集合: {}\n差集合: {}\n".format(union, intersec, diff))

if "se" in X:
    se_in_X = "yes"
else:
    se_in_X = "no"
print("seがXに含まれる: " + se_in_X)

if "se" in Y:
    se_in_Y = "yes"
else:
    se_in_Y = "no"
print("seがYに含まれる: " + se_in_Y)
