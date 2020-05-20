str = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
words = [word.strip(',.') for word in str.split()]
numbers = [1, 5, 6, 7, 8, 9, 15, 16, 19]
dict = {}

def one_char(i):
    words[i] = (words[i])[0:1]
    return

def two_char(i):
    words[i] = (words[i])[0:2]
    return

for i in range(len(words)):
    one_char(i) if ((i + 1) in numbers) else two_char(i)
    dict[words[i]] = words.index(words[i]) + 1

print(dict)

'''
dict = {}
for i in range(len(words)):
    dict.update({words[i]:i + 1})

print(dict)

values = range(1, len(words) + 1)
dict = {}
for i in range(len(words)):
    dict.update(zip(words, values))

print(dict)
'''
