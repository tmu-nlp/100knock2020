from knock03 import del_punct

sentence = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
words_fixed = del_punct(sentence)

ans = {}
for i, word in enumerate(words_fixed, start=1):
    if i in set([1, 5, 6, 7, 8, 9, 15, 16, 19]):
        word = word[:1]
    else:
        word = word[:2]
    ans[word] = i
print(ans)

