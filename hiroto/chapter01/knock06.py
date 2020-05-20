def bi_gram(str): #character
    list = []
    index = 0
    for i in range(len(str) - 1):
        list.append(str[index : index + 2])
        index += 1
    return list

s1 = 'paraparaparadise'
s2 = 'paragraph'
X, Y = set(bi_gram(s1)), set(bi_gram(s2)) #set():リストを集合にする
Union = X.union(Y) #union():和集合 |
Intersection = X.intersection(Y) #intersection():積集合 & X∩Y
Difference = X.difference(Y) #X.difference(Y):X - Y -
print('Union : ', Union)
print('Intersection : ', Intersection)
print('Difference set : ', Difference)

if 'se' in X:
    print("'se' is included in the X.")
else:
    print("'se' is not included in the X.")


if 'se' in Y:
    print("'se' is included in the Y.")
else:
    print("'se' is not included in the Y.")
