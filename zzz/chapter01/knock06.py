def n_gram(s, n, char_level=False):
    if char_level:
        inputs = [c for c in s]
    else:
        inputs = s.split(' ')

    return ['|'.join(inputs[i - n: i]) for i in range(n, len(inputs) + 1)]


if __name__ == '__main__':
    s1 = 'paraparaparadise'
    s2 = 'paragraph'

    X = set(n_gram(s1, 2, char_level=True))
    Y = set(n_gram(s2, 2, char_level=True))

    # print(X)
    # print(Y)

    print('Union:', X.union(Y))
    print('Intersection', X.intersection(Y))
    print('Difference', X.difference(Y))

    print('"se" in X:', 's|e' in X)
    print('"se" in Y:', 's|e' in Y)