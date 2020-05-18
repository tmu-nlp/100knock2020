from knock05 import create_ngram

sentence_x = 'paraparaparadise'
sentence_y = 'paragraph'

X = set(create_ngram(sentence_x, 2))
Y = set(create_ngram(sentence_y, 2))

ans_union = X | Y
ans_inter = X & Y
ans_diff_X_from_Y = X - Y
ans_diff_Y_from_X = Y - X
print('Union: {}'.format(ans_union))
print('Intersection: {}'.format(ans_inter))
print('Difference (X-Y): {}'.format(ans_diff_X_from_Y))
print('Difference (Y-X): {}'.format(ans_diff_Y_from_X))

print('"se" in X: {}'.format("se" in X))
print('"se" in Y: {}'.format("se" in Y))

