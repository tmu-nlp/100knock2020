def pull_out(i, word):
    if i in [1, 5, 6, 7, 8, 9, 15, 16, 19]:
        return (word[0], i)
    else:
        return (word[:2], i)

text = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
word = text.replace('.', '').replace(',', '')
ans = [pull_out(i, w) for i, w in enumerate(word.split(), 1)]
print(dict(ans))