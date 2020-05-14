sent = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
sent = sent.replace('.', '').split(' ')

dictAtom = {}
for num in [1, 5, 6, 7, 8, 9, 15, 16, 19]:
  dictAtom[sent[num-1][0]] = num

for num in [2, 3, 4, 10, 11, 12, 13, 14, 17, 18]:
  dictAtom[sent[num-1][:2]] = num

print(dictAtom)