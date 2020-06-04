def n_gram(target, n):
    return [target[idx:idx + n] for idx in range(len(target) - n + 1)]


text_1 = "paraparaparadise"
text_2 = "paragraph"

X = n_gram(text_1, 2)
Y = n_gram(text_2, 2)

print(f'和集合: {set(X) | set(Y)}')
print(f'積集合: {set(X) & set(Y)}')
print(f'差集合: {set(X) - set(Y)}')
print("判定: " + str('se' in (set(X) & set(Y))))