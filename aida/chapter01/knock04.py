from knock03 import del_punct

sentence = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
words_fixed = del_punct(sentence)

ans = {}
for i in range(len(words_fixed)):
    if i in set([0, 4, 5, 6, 7, 8, 14, 15, 18]):
        word = words_fixed[i][:1]
    else:
        word = words_fixed[i][:2]
    ans[word] = i
print(ans)

