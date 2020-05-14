from knock05 import char_n_gram
#how to import the function from different place?


X = char_n_gram('paraparaparadise', 2)
Y = char_n_gram('paragraph', 2)

print('X ∪ Y is ', end='')
print(X.union(Y))
print('X ∩ Y is ', end='')
print(X.intersection(Y))
print('X - Y is ', end='')
print(X.difference(Y))
print('Y - X is ', end='')
print(Y.difference(X))

print('Is \'se\' in X ? : ', end='') 
print('se' in X)
print('Is \'se\' in Y ? : ', end='') 
print('se' in Y)
