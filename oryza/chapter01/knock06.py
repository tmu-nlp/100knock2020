"""
Knock06

Let the sets of letter bi-grams from the words “paraparaparadise” and “paragraph” $X$ and $Y$, 
respectively. Obtain the union, intersection, difference of the two sets. 
In addition, check whether the bigram “se” is included in the sets $X$ and $Y$
"""
from knock05 import charNgram

w1 = 'paraparaparadise'
w2 = 'paragraph'

x = set(charNgram(w1,2))
y = set(charNgram(w2,2))

print('X: ' + str(x))
print('Y: ' + str(y))
print('X union Y: ' + str(x.union(y)))
print('X intersects Y: ' + str(x.intersection(y)))
print('X not in Y: ' + str(x.difference(y)))
print('Y is not in X: ' + str(y.difference(x)))

if 'se' in x:
    print('se is in X')
else: 
    print('se is not in X')
    
if 'se' in y:
    print('se is in Y')
else: 
    print('se is not in Y')
